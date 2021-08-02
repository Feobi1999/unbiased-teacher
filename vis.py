from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pylab, os, cv2, shutil
import os.path as osp
import numpy as np

coco_classes=[
    "__background",
    "person",
    "car",
    "train",
    "rider",
    "truck",
    "motorcycle",
    "bicycle",
    "bus"
    ]
SelectedCats=["__background",
        "person",
        "car",
        "train",
        "rider",
        "truck",
        "motorcycle",
        "bicycle",
        "bus"]

def showimg(coco, img_prefix, img, SelectedCats=None):
    I= cv2.imread(os.path.join(img_prefix,img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

def showbyplt(coco, img_prefix, img, classes, SelectedCats=None,fig=None):

    im= cv2.imread(os.path.join(img_prefix,img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    im = im[:, :, (2, 1, 0)]
    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])
    if fig is None:
        raise(fig is not None,'fig is None')
    fig.set_size_inches(width / 100, height / 100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if name in SelectedCats:
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [name, 1.0, xmin, ymin, xmax, ymax]
                objs.append(obj)
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, linewidth=1.0, color='g'))
                ax.annotate(name, (xmin,ymin), color='g',
                            weight='bold', fontsize=6, ha='center', va='center')
    print("{} find {} objects".format(img['file_name'],len(objs)))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    filename=img['file_name'].split('/')[1]
    plt.savefig(filename, dpi=100)
    # plt.show() #在plt.ion()模式下，不会阻断
    # plt.waitforbuttonpress() #阻断，等待鼠标点击或者键盘按键后继续运行
    # plt.clf()

def showbycv(coco, img_prefix, img, classes, SelectedCats=None):
    I= cv2.imread(os.path.join(img_prefix,img['file_name']))
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if name in SelectedCats:
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [name, 1.0, xmin, ymin, xmax, ymax]
                objs.append(obj)
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 0.5, (0, 0, 255))
    cv2.imshow("img", I)
    cv2.waitKey(0) #0表示等待键盘按键

def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        # print(str(cat['id'])+":"+cat['name'])
    return classes

if __name__=="__main__":
    annFile = '/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_foggy_train_cocostyle.json'
    img_prefix='/media/sda2/mzhe/datasets/Cityscapes/leftImg8bit_foggy/train'
    coco = COCO(annFile)
    coco_catid2name = catid2name(coco)
    fig = plt.figure()
    plt.ion()  # matplotlib interactivate mode 当交互模式打开后，plt.show()不会阻断运行
    for img_id in coco.imgs:
        img=coco.imgs[img_id]
        # showimg(coco,img_prefix,img,SelectedCats)
        # 通多opencv交互显示图像和标注；按下键盘回车或者空格自动显示下一张图像
        # showbycv(coco,img_prefix,img,coco_catid2name,SelectedCats)
        # 通过matplotlib.pylot交互显示图像和标注；点击鼠标或者键盘自动显示下一张图像
        showbyplt(coco, img_prefix, img, coco_catid2name, SelectedCats,fig=fig)
