from flask import Flask, request, render_template
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from tensorflow import keras


app = Flask(__name__)

# model = keras.models.load_model('my_model')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def prediction():

    data = request.form.get('url')

    # data = data.split(',')[1]

    offset = data.index(',')+1
    img_bytes = base64.b64decode(data[offset:])
    img = Image.open(BytesIO(img_bytes))
    img  = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28,28))
    img = img.reshape(1,28,28,1)
    image = img/255

    model = keras.models.load_model('my_model')
    result = model.predict(image).argmax()


    # cv2.imshow("image",img)
    # cv2.waitKeys(0)
    # cv2.destroyAllWindows()



    # nparr = np.fromstring(data, np.uint8)

    # imgstr = re.search(b'data:image/png;base64,(.*)', nparr)
    # with open('output.png', 'wb') as output:
    #     output.write(base64.b64decode(imgstr))



    # imgstr = re.search(b'base64,(.*)', data).group(1)
    # with open('output.png', 'wb') as output:
    #     output.write(base64.decodebytes(data))

    # x = imread('outmut.png', mode='L')

    # x = np.invert(x)

    # x = cv2.resize(x, (28,28))
    # # x = imresize(x, (28,28))
    # x = x.reshape(1,28,28,1)
    

    # out = model.predict(x).argmax()


    # encoded_data = data.split(',')[1]

    # nparr = np.fromstring(encoded_data, np.uint8)
    # list_img = list(nparr)
    # img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


    # resize_img = cv2.resize(nparr, (28,28))

    # image = resize_img.reshape(1,28,28,1)

    # image = image/255

    # model = keras.models.load_model('my_model')

    # result = model.predict(image)

    return render_template('index.html', result=result)


if(__name__ == '__main__'):
    app.run(debug=True)