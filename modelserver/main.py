"""
Model server using ResNet50 classier and Redis to poll for imagess

"""

import base64
import json
import os
import sys
import time

from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import redis

#Connect to redis server
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

#Load pretrained model
model = ResNet50(weights=None)
model.load_weights("model/resnet50_weights_tf_dim_ordering_tf_kernels.h5")	

def base64_decode_image(a, shape):
	#Encoding the serialized Numpy string as a byte object for Python 3
	if sys.version_info.major == 3:
		a = bytes(a,encoding="utf-8")

	#Convert the string to a Numpy array using the supplied data type and shape
	a = np.frombuffer(base64.decodestring(a),dtype=np.uint8)
	print(a.shape)
	a = a.reshape(shape)

	return a

def classify_process():
	#Poll for new images to classify
	while True:
		#Pop images from Redis queue atomically
		with db.pipeline() as pipe:
			pipe.lrange(os.environ.get("IMAGE_QUEUE"), 0, int(os.environ.get("BATCH_SIZE"))-1)
			pipe.ltrim(os.environ.get("IMAGE_QUEUE"), int(os.environ.get("BATCH_SIZE")),-1)
			queue, _ = pipe.execute()

		imageIDs = []
		batch = None

		for q in queue:
			q = json.loads(q.decode("utf-8"))
			image = base64_decode_image(q["image"],
                                        (1, int(os.environ.get("IMAGE_HEIGHT")),
                                         int(os.environ.get("IMAGE_WIDTH")),
                                         int(os.environ.get("IMAGE_CHANNEL")))
                                        )
			image = imagenet_utils.preprocess_input(image)
			if batch is None:
				batch = image
			else:
				batch = np.vstack([batch,image])

			imageIDs.append(q["id"])

		if len(imageIDs) > 0:

			print("Batch Size: {}".format(batch.shape))
			pred = model.predict(batch)
			results = imagenet_utils.decode_predictions(pred)

			for (imageID, resultSet) in zip(imageIDs,results):
				output = []
				for (imagenetID, label, prob) in resultSet:
					r = {"label":label, "probability":float(prob)}
					output.append(r)
				db.set(imageID, json.dumps(output))


		#Small Sleep
		time.sleep(float(os.environ.get("SERVER_SLEEP")))


if __name__ == "__main__":
	classify_process()
