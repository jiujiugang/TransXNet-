from os import path
import os
import cv2
import time
import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch
from crossformer import CrossFormer
import numpy as np
# Some of the codes are adapted from STSNet
import warnings
warnings.filterwarnings("ignore")
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #             print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):  #这段代码定义了一个函数 confusionMatrix，用于计算混淆矩阵的相关指标，包括 F1 分数和平均召回率
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):#定义一个函数 recognition_evaluation，它接受三个参数：final_gt（真实的情感标签），final_pred（模型预测的情感标签），以及一个可选参数 show
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}#情感标签映射到数字编码
    # Display recognition result
    f1_list = []
    ar_list = []#初始化了两个空列表，用于存储每个情感类别的F1分数和识别率。
    try:
        for emotion, emotion_index in label_dict.items():#遍历 label_dict 字典的键值对，emotion 表示情感标签的字符串形式，emotion_index 表示对应的数值形式。
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]#使用列表推导式生成两个新列表 gt_recog 和 pred_recog
        #对于 final_gt 和 final_pred 中的每个元素，如果它等于当前的 emotion_index，则在新列表中标记为 1（表示该样本属于当前情感类别），否则标记为 0。
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)#将计算得到的F1分数和识别率添加到相应的列表中。
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR#计算所有情感类别的平均F1分数和平均识别率。
    except:
        return '', ''#如果发生异常，返回两个空字符串。

# 1. get the whole face block coordinates
def whole_face_block_coordinates():
    df = pandas.read_csv('../dataset/data/442.csv')
    m, n = df.shape
    base_data_src = '../dataset/data/apex'
    total_emotion = 0
    image_size_u_v = 224

    # 存储图像路径的字典
    face_block_coordinates = {}

    # 遍历CSV文件中的每一行
    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(df['filename_o'][i]) + '.jpg'  # 生成图像名称
        img_path_apex = base_data_src + '/' + df['imagename'][i]  # 构建图像路径
        train_face_image_apex = cv2.imread(img_path_apex)  # 读取图像

        # 将图像名称与图像路径存储在字典中
        face_block_coordinates[image_name] = img_path_apex

    return face_block_coordinates

# 2. crop the 28*28-> 14*14 according to i5 image centers
def crop_optical_flow_block():
    # 其他部分保持不变
    whole_optical_flow_path = '../dataset/data/flow'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)

    four_parts_optical_flow_imgs = {}
    for n_img in whole_optical_flow_imgs:
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        # 这里假设flow_image是一个包含光流数据的图像
        four_parts_optical_flow_imgs[n_img] = flow_image  # 直接保存光流图像
    return four_parts_optical_flow_imgs


def crop_apex_frame_block():
    # 顶点帧数据路径
    apex_frame_path = r'C:\Users\24291\Desktop\CloFormer-main\dataset\data\apex'
    whole_apex_frame_imgs = os.listdir(apex_frame_path)

    apex_frames = {}
    for n_img in whole_apex_frame_imgs:
        # 读取顶点帧图像
        apex_image = cv2.imread(os.path.join(apex_frame_path, n_img))
        if apex_image is not None:
            apex_frames[n_img] = apex_image  # 使用文件名作为键保存顶点帧图像
        else:
            print(f"Warning: {n_img} could not be loaded.")  # 打印无法加载的图像文件名

    return apex_frames


def main(config):
    learning_rate = 0.00005
    batch_size = 32
    epochs = 40
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss()
    if (config.train):
        if not path.exists('ourmodel_threedatasets_weightsDL'):
            os.mkdir('ourmodel_threedatasets_weightsDL')#检查是否存在保存模型权重的文件夹，如果不存在则创建。

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []#初始化三个空列表，可能用于存储真实标签、预测结果和最佳预测结果。

    t = time.time()#获取当前时间（以秒为单位），可能是为了计算训练或测试过程所需的时间

    main_path = '../dataset/data/loso'
    #apex_frame_path = r'C:\Users\24291\Desktop\CloFormer-main\dataset\data\apex_loso'  # 新增：顶点帧数据路径
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()#调用了crop_optical_flow_block函数，获取了五个局部光流图像，并将其存储在all_five_parts_optical_flow变量中。
    apex_data = crop_apex_frame_block()  # 加载顶点帧数据，不再需要传递路径
  # 新增：加载顶点帧数据
    print(subName)

    for n_subName in subName:#遍历主数据集下的每个子文件夹
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []
        apex_frames_train = []  # 新增：顶点帧训练数据
        apex_frames_test = []  # 新增：顶点帧测试数据
        # Get train dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')#获取当前子目录下u_train文件夹中的所有文件和子目录名，并将它们存储在expression列表中
        for n_expression in expression:#遍历训练集文件夹下的每个表情类别
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)#获取当前表情类别下的图像文件名列表。

            for n_img in img:
                y_train.append(int(n_expression))
                # 直接从光流字典中获取光流数据
                flow_data = all_five_parts_optical_flow[n_img]  # 假设flow_data是(28, 28, 3)的图像
                four_parts_train.append(flow_data)
                # 获取对应的顶点帧数据
                apex_frame = apex_data[n_img]#同样使用图像文件名 n_img，从 apex_data 字典中获取对应的顶点帧数据
                apex_frames_train.append(apex_frame)

        # Get test dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                flow_data = all_five_parts_optical_flow[n_img]
                four_parts_test.append(flow_data)
                # 获取对应的顶点帧数据
                apex_frame = apex_data[n_img]
                apex_frames_test.append(apex_frame)
        weight_path = 'ourmodel_threedatasets_weightsDL' + '/' + n_subName + '.pth'

        # Reset or load model weigts
        model = CrossFormer(num_classes=3, fusion_method='concat')
        # 测试模型输出
        flow_data = torch.randn(1, 3, 224, 224)
        vertex_data = torch.randn(1, 3, 224, 224)
        output = model(flow_data, vertex_data)
        print("Output shape:", output.shape)


        model = model.to(device)

        if(config.train):
            # model.apply(reset_weights)
            print('train')
        else:
            model.load_state_dict(torch.load(weight_path))
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)#将训练标签转换为长整型Tensor
        four_parts_train =torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)#将训练数据从numpy数组转换为Tensor，并调整其维度顺序。
        apex_frames_train = torch.Tensor(np.array(apex_frames_train)).permute(0, 3, 1, 2)  # 确保维度正确
        dataset_train = TensorDataset(four_parts_train, apex_frames_train, y_train)  # 包装光流和顶点帧

        train_dl = DataLoader(dataset_train, batch_size=batch_size)#创建一个DataLoader，用于在训练时按批次提供数据。

        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        apex_frames_test = torch.Tensor(np.array(apex_frames_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, apex_frames_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)#这部分代码与准备训练数据的步骤类似，只是它是为测试数据准备的
        # store best results
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        for epoch in range(1, epochs + 1):
            if (config.train):
                # Training
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    flow_data = batch[0].to(device)  # 光流数据
                    vertex_data = batch[1].to(device)  # 顶点帧数据
                    y = batch[2].to(device)  # 标签
                    yhat = model(flow_data, vertex_data)  # 传递两个输入
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * flow_data.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += flow_data.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)

            # Testing
            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                flow_data = batch[0].to(device)
                vertex_data = batch[1].to(device)
                y = batch[2].to(device)
                yhat = model(flow_data, vertex_data)
                loss = loss_fn(yhat, y)

                val_loss += loss.data.item() * flow_data.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)

            #### best result
            temp_best_each_subject_pred = []#一个空列表，用于暂时存储每个子主题的最佳预测结果。
            if best_accuracy_for_each_subject <= val_acc:#它检查当前的验证准确率是否大于等于之前记录的最佳准确率。
                best_accuracy_for_each_subject = val_acc#如果当前的验证准确率更好，那么更新最佳准确率为当前准确率。
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())#这一行代码将当前批次的预测结果（yhat）中每个样本的最大概率对应的类别索引提取出来，并加入到 temp_best_each_subject_pred 列表中。
                best_each_subject_pred = temp_best_each_subject_pred#更新记录每个子主题的最佳预测结果
                # Save Weights
                if (config.train):
                    torch.save(model.state_dict(), weight_path)#如果是，在达到新的最佳准确率时，就保存当前模型的权重到指定的路径中


        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y.tolist()
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')

        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)




if __name__ == '__main__':
    # get_whole_u_v_os()
    # create_norm_u_v_os_train_test()
    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--train', type=strtobool, default=True)  # Train or use pre-trained weight for prediction
    config = parser.parse_args()
    main(config)