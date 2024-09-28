import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.local.multi_distances import calculate_multi_distance

class QdrantEngine:
    def __init__(self, url, image_dim: int, audio_dim: int, text_dim: int):
        self.client = QdrantClient(url)
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim

        if not self.client.collection_exists(collection_name="videos"):
            self._create(self.image_dim, self.audio_dim, self.text_dim)

    def _create(self, image_dim: int, audio_dim: int, text_dim: int):
        self.client.create_collection(
            collection_name="videos",
            vectors_config={
                "image": models.VectorParams(
                    size=image_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "audio": models.VectorParams(
                    size=audio_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                "text": models.VectorParams(
                    size=text_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
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

    #0.65 for image, 0.95 for audio, 0.85 for text
    def search(self, image_embedding, audio_embedding=None, text_embedding=None, top_k=10, image_threshold=0.65, audio_threshold=0.95, text_threshold=0.95):
        img_q = {
            "query": image_embedding,
            "using": "image",
            "score_threshold": image_embedding.shape[0] * image_threshold,
            "limit": top_k * 10,
        }
        if audio_embedding is not None:
            audio_q = {
                "query": audio_embedding,
                "using": "audio",
                "score_threshold": audio_embedding.shape[0] * audio_threshold,
                "limit": top_k * 5
            }
        if text_embedding is not None:
            text_q = {
                "query": text_embedding,
                "using": "text",
                "score_threshold": text_embedding.shape[0] * text_threshold,
                "limit": top_k
            }
        if audio_embedding is None and text_embedding is None:
            results = self.client.query_points(
                collection_name="videos",
                **img_q,
                with_vectors=True
            )
        elif text_embedding is None:
            results = self.client.query_points(
                collection_name="videos",
                prefetch=models.Prefetch(**img_q),
                **audio_q,
                with_vectors=True
            )
        else:
            results = self.client.query_points(
                collection_name="videos",
                prefetch=models.Prefetch(**audio_q, prefetch=models.Prefetch(**img_q)),
                **text_q,
                with_vectors=True
            )
        # results = self.client.query_points(
        #     collection_name="videos",
        #     prefetch=[
        #       models.Prefetch(
                
        #       ),
        #       models.Prefetch(
        #         query=audio_embedding,
        #         using="audio",
        #         threshold=self.audio_threshold,
        #         limit=top_k * 5
        #       ),
        #     ],
        #     query=text_embedding,
        #     using="text",
        #     threshold=self.text_threshold,
        #     limit=top_k
        # )
        return results
    

def score(embedding1, embedding2, distance_type="Cosine"):
    if type(embedding1) == list:
        embedding1 = np.array(embedding1)
    if type(embedding2) == list:
        embedding2 = np.array(embedding2)
    return calculate_multi_distance(embedding1, np.array([embedding2]), distance_type=distance_type)[0]