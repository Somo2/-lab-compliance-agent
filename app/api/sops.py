from fastapi import APIRouter, HTTPException

from app.retrieval.sop_loader import SOPLoader
from app.utils.normalization import normalize_sop_id


router = APIRouter(prefix="/sops", tags=["SOPs"])


@router.get("")
def list_sops() -> list[dict]:
    """List all available SOPs."""
    documents = SOPLoader().load()

    sop_ids = sorted({document.sop_id for document in documents})

    return [
        {
            "sop_id": sop_id,
        }
        for sop_id in sop_ids
    ]


@router.get("/{sop_id}")
def get_sop(sop_id: str) -> dict:
    """Return the sections and metadata for a specific SOP."""
    normalized_sop_id = normalize_sop_id(sop_id)
    documents = SOPLoader().load()

    matching_documents = [
        document
        for document in documents
        if document.sop_id.upper() == normalized_sop_id
    ]

    if not matching_documents:
        raise HTTPException(
            status_code=404,
            detail=f"SOP {normalized_sop_id} not found",
        )

    return {
        "sop_id": matching_documents[0].sop_id,
        "sections": [
            {
                "section": document.section,
                "content": document.content,
                "metadata": document.metadata,
            }
            for document in matching_documents
        ],
    }
