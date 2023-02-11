import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms import functional as F
# import pytesseract
# import pyttsx3

from PyQt6 import QtWidgets
from PyQt6 import QtGui
from PyQt6 import QtCore
from PyQt6.QtCore import Qt

def qimage_to_numpy(qimage):
    qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.constBits()
    ptr.setsize(height * width * 3)
    return np.frombuffer(ptr, np.uint8).reshape((height, width, 3))

def get_bboxes_from_model(model, image):
    image = Image.fromarray(image, 'RGB')
    image_tensor = F.to_tensor(image)
    with torch.no_grad():
        output = model([image_tensor])
    return output[0]['boxes'].numpy()

def mask_to_contours(mask):
    return find_contours(mask, 0.5)

def crop_to_text(img_crop):
    # first try with original image
    text1 = pytesseract.image_to_string(img_crop)
    print('text1: ', text1)

    img_crop = rgb2gray(img_crop)

    # global thresholding
    thresh = threshold_otsu(img_crop)
    img_crop_t = img_crop > thresh
    text2 = pytesseract.image_to_string(img_crop_t)
    print('text2: ', text2)

    # the best so far
    text = text1 if len(text1) > len(text2) else text2

    # local thresholding
    thresh = threshold_local(img_crop, block_size=35, offset=10)
    img_crop_t = img_crop > thresh
    text3 = pytesseract.image_to_string(img_crop_t)
    print('text3: ', text3)

    # the best
    return text if len(text) > len(text3) else text3

def contour_to_crop(img, contour):
    # mask from contour
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [(point[1], point[0]) for point in contour]
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)

    # apply mask to image
    img_mask = np.zeros_like(img)
    img_mask[mask] = img[mask]
    Image.fromarray(img_mask, 'RGB').save('crop_mask.png')

    # crop
    (y, x) = np.where(mask)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    img_crop = img[topy:bottomy+1, topx:bottomx+1]
    #print(img_crop.shape)
    Image.fromarray(img_crop, 'RGB').save('crop.png')
    return img_crop

def contour_to_text(img, contour):
    img_crop = contour_to_crop(img, contour)
    return crop_to_text(img_crop)

def get_mask_from_model(model, image, threshold = 0.5):
    image = Image.fromarray(image, 'RGB')
    image_tensor = F.to_tensor(image)
    output = torch.sigmoid(model(image_tensor.unsqueeze(0))['out'])
    output = (output > threshold).squeeze(0)
    return output[0, :, :].numpy() # only text at this point

def get_model(model_name, num_classes, checkpoint):
    model = torch.load(checkpoint, map_location='cpu')
    model.model.eval()
    model.eval()
    return model
    # model = torchvision.models.get_model(model_name, weights=None, weights_backbone=None, num_classes=num_classes)
    # checkpoint = torch.load(checkpoint, map_location='cpu')
    # model.load_state_dict(checkpoint)
    # model.eval()
    # return model

class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.scale = 1.0
        self.text = ''

        layout = QtWidgets.QGridLayout(self)

        self.select = QtWidgets.QPushButton('Select')
        layout.addWidget(self.select)
        self.select.clicked.connect(self.load_pixmap_from_filename)
        
        self.label = QtWidgets.QLabel()
        layout.addWidget(self.label)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.label.installEventFilter(self)
        # self.label.setScaledContents(True)
        # self.label.setFixedSize(800, 600)

        # load the model
        self.model = get_model(model_name='fasterrcnn_resnet50_fpn', num_classes=3, checkpoint='model.pth')

        # action for pasteing image from clipboard
        self.action = QtGui.QAction(('Paste'), self)
        self.action.setShortcut(QtGui.QKeySequence("Ctrl+v"))
        self.addAction(self.action)
        self.action.triggered.connect(self.load_pixmap_from_clipboard)

        # action for copying the last selected text into the clipboard
        # self.action2 = QtWidgets.QAction(('Copy'), self)
        # self.action2.setShortcut(QtGui.QKeySequence("Ctrl+c"))
        # self.addAction(self.action2)
        # self.action2.triggered.connect(self.save_text_to_clipboard)

    # def save_text_to_clipboard(self):
    #     QtWidgets.QApplication.clipboard().setText(self.text)

    def load_pixmap_from_clipboard(self):
        pixmap = QtWidgets.QApplication.clipboard().pixmap()
        if pixmap:
            self.pixmap = pixmap #.scaledToWidth(1280)
            self.load_pixmap()

    def load_pixmap_from_filename(self):
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(None, 'Resim Yükle', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)')
        if not filename:
            return
        self.pixmap = QtGui.QPixmap(filename)
        self.load_pixmap()

    def load_pixmap(self):
        print('loading ...')
        self.image = qimage_to_numpy(self.pixmap.toImage())
        self.bboxes = get_bboxes_from_model(self.model, self.image)
        self.draw_bboxes()
        self.label.setPixmap(self.pixmap)
        print('done')

    def draw_bboxes(self):
        painter = QtGui.QPainter(self.pixmap)
        painter.drawPixmap(self.pixmap.rect(), self.pixmap)
        pen = QtGui.QPen(QtGui.QColor(0, 0, 255), 3)
        painter.setPen(pen)
        # painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.bboxes.shape[0] <= 0:
            print('no bboxes detected')
            return
        for i in range(self.bboxes.shape[0]):
            painter.drawRect(self.bboxes[i][0], self.bboxes[i][1], self.bboxes[i][2]-self.bboxes[i][0], self.bboxes[i][3]-self.bboxes[i][1])
        painter.end()

    def draw_contours(self):
        painter = QtGui.QPainter(self.pixmap)
        painter.drawPixmap(self.pixmap.rect(), self.pixmap)
        pen = QtGui.QPen(Qt.blue, 1)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for contour in self.contours:
            for i in range(contour.shape[0]):
                painter.drawPoint(int(contour[i, 1]), int(contour[i, 0]))
        painter.end()

    def eventFilter(self, source, event):
        # if the source is our QLabel, it has a valid pixmap, and the event is
        # a left click, proceed in trying to get the event position
        if (source == self.label and 
            source.pixmap() and 
            not source.pixmap().isNull() and 
            event.type() == QtCore.QEvent.MouseButtonPress and
            event.button() == QtCore.Qt.LeftButton
        ):
            self.handle_mouse_click_event(event)
        return super().eventFilter(source, event)

    def handle_mouse_click_event(self, event):
        pos = self.transformPos(event.localPos())
        print('X={}, Y={}'.format(pos.x(), pos.y()))
        for contour in self.contours:
            if points_in_poly([(int(pos.y()), int(pos.x()))], contour):
                import winsound         
                winsound.Beep(600, 250)
                self.text = contour_to_text(self.image, contour)
                print(self.pixmap)
                print(self.text)
                pyttsx3.speak(self.text)

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = self.label.contentsRect()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) # if aw > w else 0
        y = (ah - h) / (2 * s) # if ah > h else 0
        return QtCore.QPoint(x, y)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec())
