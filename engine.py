from unittest import result
from qdrant_client import QdrantClient, models

class QdrantEngine:
    def __init__(self, url, image_dim: int, audio_dim: int, text_dim: int, image_threshold=0.75, audio_threshold=0.75, text_threshold=0.95):
        self.client = QdrantClient(url)
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.image_threshold = image_threshold
        self.audio_threshold = audio_threshold
        self.text_threshold = text_threshold

        self._create(self.image_dim, self.audio_dim, self.text_dim)

    def _create(self, image_dim: int, audio_dim: int, text_dim: int):
        self.client.create_collection(
            collection_name="videos",
            vectors_config={
                "image": models.VectorParams(
                    size=image_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorCompator.MAX_SIMILARITY
                    ),
                ),
                "audio": models.VectorParams(
                    size=audio_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorCompator.MAX_SIMILARITY
                    ),
                ),
                "text": models.VectorParams(
                    size=text_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorCompator.MAX_SIMILARITY
                    ),
                ),
            }
        )

    def add(self, video_id, image_embedding, audio_embedding, text_embedding):
        self.client.upsert(
            collection_name="videos",
            points=[
                models.PointStruct(
                    id=video_id,
                    vector={
                        "image": image_embedding,
                        "audio": audio_embedding,
                        "text": text_embedding
                    }
                )
            ],
        )

    
    def search(self, image_embedding, audio_embedding, text_embedding, top_k=10):
        results = self.client.query_points(
            collection_name="videos",
            prefetch=[
              models.Prefetch(
                query=image_embedding,
                using="image",
                threshold=self.image_threshold,
                limit=top_k * 10
              ),
              models.Prefetch(
                query=audio_embedding,
                using="audio",
                threshold=self.audio_threshold,
                limit=top_k * 5
              ),
            ],
            query=text_embedding,
            using="text",
            threshold=self.text_threshold,
            limit=top_k
        )
        return results