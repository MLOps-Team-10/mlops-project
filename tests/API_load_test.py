import random
import os
from locust import HttpUser, between, task


class MyUser(HttpUser):
    """A simple Locust user class that simulates the tasks performed by users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(9)
    def post_image(self) -> None:
        """A task that simulates a user uploading an image for classification."""

        image_path = "data/raw/eurosat_rgb/Highway/Highway_1.jpg"

        with open(image_path, "rb") as image_file:
            files = {"file": ("Highway_1.jpg", image_file, "image/jpeg")}
            self.client.post("/takeaguess", files=files)
