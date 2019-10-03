import base64
import flask
import redis
import time
import json
import uuid
import os
from PIL import Image
import io
import numpy as np

app = flask.Flask(__name__)
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = 100 #int(os.environ.get("CLIENT_MAX_TRIES"))

def preprocess_image(image, target):
    # If the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the input image and preprocess it
    image = image.resize(target)
    image =  np.array(image)
    image = np.expand_dims(image, axis=0)
    # Return the processed image
    return image	

@app.route("/")
def homepage():
	return "Hello World"

@app.route("/predict", methods=["POST"])
def predict():

	data = {"success":False}

	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = preprocess_image(image,(int(os.environ.get("IMAGE_HEIGHT")), int(os.environ.get("IMAGE_WIDTH"))))
			# Ensure our NumPy array is C-contiguous as well, otherwise we won't be able to serialize it

			image = image.copy(order="C")
			image = base64.b64encode(image).decode("utf-8")
			
			imgId = str(uuid.uuid4())
			d = {"id":imgId, "image": image}
			db.rpush(os.environ.get("IMAGE_QUEUE"),json.dumps(d))

			num_tries = 0
			while num_tries < CLIENT_MAX_TRIES:
				num_tries += 1
				output = db.get(imgId)

				if output is not None:
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					db.delete(imgId)
					break

				time.sleep(float(os.environ.get("CLIENT_SLEEP")))

			data["success"] = True

	return flask.jsonify(data)


if __name__ == "__main__":
	print("* Starting web service...")
	app.run(host='0.0.0.0')