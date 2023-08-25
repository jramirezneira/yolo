from flask import Flask, render_template
from ultralytics import YOLO

# template_folder points to current directory. Flask will look for '/static/'
app = Flask(__name__, template_folder='.')
# The rest of your file here

@app.route('/')
def index():
  """ Serving static files """
  try:
    #return render_template('index.html')
    model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
    #model.train(data='coco128.yaml', epochs=3)  # train the model
    results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image



    return results[0].tojson()

    """     for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
        print(boxes)
        print(masks)
        print(probs) """

  except:
    return str(e)
if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True, port=5000)
