import os
import asyncio
import json
import logging
import shutil
from pathlib import Path

import boto3
import httpx
from aio_pika import connect_robust, IncomingMessage
from botocore.config import Config
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient, models

from app.ollama_client import generate_embeddings
from app.sparse import encode_sparse

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MINERU_TIMEOUT = 600

# ==========================================
# 1. STRICT ENVIRONMENT ENFORCEMENT
# ==========================================
try:
    AMQP_URL = os.environ["AMQP_URL"]
    R2_ENDPOINT_URL = os.environ["R2_ENDPOINT_URL"]
    R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
    R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
    R2_BUCKET_NAME = os.environ["R2_BUCKET_NAME"]
    QDRANT_URL = os.environ["QDRANT_URL"]
    QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
    JAVA_BACKEND_CALLBACK_URL = os.environ["JAVA_BACKEND_CALLBACK_URL"]

    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
except KeyError as e:
    logger.critical("FATAL BOOT ERROR: Missing environment variable %s", e)
    exit(1)

_http_client: httpx.AsyncClient | None = None


def _get_http() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


# ==========================================
# 2. R2 DOWNLOAD (async wrapper around sync boto3)
# ==========================================


def _s3_download(key: str, dest: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )
    client.download_file(R2_BUCKET_NAME, key, dest)


async def download_from_r2(s3_key: str, local_destination: str) -> None:
    await asyncio.to_thread(_s3_download, s3_key, local_destination)
    logger.info(
        "Downloaded s3_key=%s -> %s (%d bytes)",
        s3_key, local_destination,
        Path(local_destination).stat().st_size,
    )


# ==========================================
# 3. MINERU EXTRACTION (async subprocess)
# ==========================================


async def extract_with_mineru(pdf_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "magic-pdf",
        "-p", pdf_path,
        "-o", output_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=MINERU_TIMEOUT
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"MinerU timed out after {MINERU_TIMEOUT}s for {pdf_path}"
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"MinerU exited code {proc.returncode}: "
            f"{stderr.decode(errors='replace')[:2000]}"
        )

    # MinerU writes output to {output_dir}/{pdf_stem}/result.md
    pdf_stem = Path(pdf_path).stem
    md_path = Path(output_dir) / pdf_stem / "result.md"
    if not md_path.is_file():
        raise RuntimeError(f"MinerU produced no result.md at {md_path}")
    return md_path.read_text(encoding="utf-8")


# ==========================================
# 4. THE CONSUMPTION LIFECYCLE
# ==========================================
async def process_message(message: IncomingMessage):
    async with message.process(requeue=False, ignore_processed=True):
        document_id = "<unknown>"
        local_pdf_path = None
        mineru_output_dir = None
        try:
            payload = json.loads(message.body.decode())
            document_id = payload.get("documentId", document_id)
            s3_key = payload.get("s3ObjectKey")

            if not document_id or not s3_key:
                raise ValueError(
                    "Invalid payload: missing documentId or s3ObjectKey"
                )

            logger.info(
                "Processing document_id=%s s3_key=%s",
                document_id, s3_key,
            )

            # ------------------------------------------------------------------
            # Step 1: Download PDF from R2 / MinIO
            # ------------------------------------------------------------------
            local_pdf_path = f"/tmp/{document_id}.pdf"
            mineru_output_dir = f"/tmp/{document_id}_output"
            await download_from_r2(s3_key, local_pdf_path)

            # ------------------------------------------------------------------
            # Step 2: MinerU extraction (GPU-heavy subprocess)
            # ------------------------------------------------------------------
            markdown = await extract_with_mineru(local_pdf_path, mineru_output_dir)
            logger.info(
                "Extracted markdown document_id=%s chars=%s",
                document_id, len(markdown),
            )

            # ------------------------------------------------------------------
            # Step 3: Generate dense + sparse vectors
            # ------------------------------------------------------------------
            dense_vector = await generate_embeddings(markdown)
            sparse_vector = encode_sparse(markdown)
            logger.info(
                "Generated vectors document_id=%s dense_dim=%s sparse_terms=%s",
                document_id, len(dense_vector), len(sparse_vector["indices"]),
            )

            # ------------------------------------------------------------------
            # Step 4: Upsert hybrid vectors to Qdrant Cloud
            # ------------------------------------------------------------------
            async with AsyncQdrantClient(
                url=QDRANT_URL, api_key=QDRANT_API_KEY,
            ) as qdrant:
                await qdrant.upsert(
                    collection_name="documents",
                    points=[
                        models.PointStruct(
                            id=document_id,
                            vector={
                                "dense": dense_vector,
                                "sparse": sparse_vector,
                            },
                            payload={
                                "document_id": document_id,
                                "markdown": markdown,
                            },
                        )
                    ],
                )
            logger.info("Upserted vectors to Qdrant document_id=%s", document_id)

            # ================================================================
            # 5. STRICT ACKNOWLEDGMENT ORDERING
            # ================================================================
            # ACK ONLY after Qdrant confirms the upsert.
            await message.ack()
            logger.info("ACKed document_id=%s", document_id)

            # ------------------------------------------------------------------
            # 6. Notify Java backend (best-effort after ACK)
            # ------------------------------------------------------------------
            client = _get_http()
            await client.post(
                JAVA_BACKEND_CALLBACK_URL,
                json={"documentId": document_id, "status": "READY"},
            )

        except Exception:
            logger.exception("Failed to process document_id=%s", document_id)
            try:
                await message.nack(requeue=False)
            except Exception:
                logger.exception("Failed to NACK document_id=%s", document_id)

            if document_id != "<unknown>":
                try:
                    client = _get_http()
                    await client.post(
                        JAVA_BACKEND_CALLBACK_URL,
                        json={"documentId": document_id, "status": "FAILED"},
                    )
                except Exception:
                    logger.exception(
                        "Failed to send FAILED webhook for %s", document_id
                    )

        finally:
            # Critical disk cleanup — always runs
            if local_pdf_path and os.path.isfile(local_pdf_path):
                os.remove(local_pdf_path)
            if mineru_output_dir and os.path.isdir(mineru_output_dir):
                shutil.rmtree(mineru_output_dir, ignore_errors=True)


# ==========================================
# 5. DAEMON ENTRYPOINT
# ==========================================
async def main():
    logger.info("Starting Evidence Pilot GPU Worker...")

    connection = await connect_robust(
        AMQP_URL,
        heartbeat=300,
        timeout=30,
        reconnect_interval=5,
    )

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        queue = await channel.declare_queue(
            "extraction.queue",
            durable=True,
            arguments={
                "x-dead-letter-exchange": "extraction.dlx",
                "x-dead-letter-routing-key": "extraction.dlq",
            },
        )

        logger.info("Bound to queue=extraction.queue. Waiting for ExtractionRequests...")
        await queue.consume(process_message)

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
