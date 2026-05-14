import math
import torch
import torch.nn as nn
from net.restormer import TransformerBlock as Restormer
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)

        return attn_output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x.permute(0, 2, 1))).permute(0, 2, 1)
        max_out = self.fc(self.max_pool(x.permute(0, 2, 1))).permute(0, 2, 1)
        out = avg_out + max_out
        # out = avg_out
        return self.sigmoid(out)


class imagefeature2textfeature(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim):
        super(imagefeature2textfeature, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.conv(x)

        x = F.interpolate(x, [288, 384], mode='nearest')
        x = x.contiguous().view(x.size(0), x.size().numel() // x.size(0) // self.hidden_dim, self.hidden_dim)
        return x


class restormer_cablock(nn.Module):
    def __init__(
            self,
            input_channel=1,
            restormerdim=32,
            restormerhead=8,
            image2text_dim=10,
            ffn_expansion_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            hidden_dim=768,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        self.convA1 = nn.Conv2d(input_channel, restormerdim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluA1 = nn.PReLU()
        self.convA2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.preluA2 = nn.PReLU()
        self.convA3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.preluA3 = nn.PReLU()

        self.convB1 = nn.Conv2d(input_channel, restormerdim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluB1 = nn.PReLU()
        self.convB2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.preluB2 = nn.PReLU()
        self.convB3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.preluB3 = nn.PReLU()

        self.image2text_dim = image2text_dim
        self.restormerA1 = Restormer(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.restormerB1 = Restormer(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.cross_attentionA1 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.cross_attentionA2 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.imagef2textfA1 = imagefeature2textfeature(restormerdim, image2text_dim, hidden_dim)
        self.imagef2textfB1 = imagefeature2textfeature(restormerdim, image2text_dim, hidden_dim)
        self.image2text_dim = image2text_dim

    def forward(self, imageA, imageB, text):
        if len(imageA.shape) == 3:
            imageA = imageA.cuda().unsqueeze(0).permute(0, 3, 1, 2)
            imageB = imageB.cuda().unsqueeze(0).permute(0, 3, 1, 2)
        b, _, H, W = imageA.shape

        imageA = self.restormerA1(self.preluA1(self.convA1(imageA)))
        imageAtotext = self.imagef2textfA1(imageA)
        imageB = self.restormerB1(self.preluB1(self.convB1(imageB)))
        imageBtotext = self.imagef2textfB1(imageB)

        ca_A = self.cross_attentionA1(text, imageAtotext, imageAtotext)
        imageA_sideout = imageA
        ca_A = torch.nn.functional.adaptive_avg_pool1d(ca_A.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_A = F.normalize(ca_A, p=1, dim=2)

        ca_A = (imageAtotext * ca_A).view(imageA.shape[0], self.image2text_dim, 288, 384)
        imageA_sideout = F.interpolate(imageA_sideout, [H, W], mode='nearest')
        ca_A = F.interpolate(ca_A, [H, W], mode='nearest')
        ca_A = self.preluA3(
            self.convA3(torch.cat(
                (F.interpolate(imageA, [H, W], mode='nearest'), self.preluA2(self.convA2(ca_A)) + imageA_sideout), 1)))

        ca_B = self.cross_attentionA2(text, imageBtotext, imageBtotext)
        imageB_sideout = imageB
        ca_B = torch.nn.functional.adaptive_avg_pool1d(ca_B.permute(0, 2, 1), 1).permute(0, 2, 1)
        ca_B = F.normalize(ca_B, p=1, dim=2)

        ca_B = (imageBtotext * ca_B).view(imageA.shape[0], self.image2text_dim, 288, 384)
        imageB_sideout = F.interpolate(imageB_sideout, [H, W], mode='nearest')
        ca_B = F.interpolate(ca_B, [H, W], mode='nearest')
        ca_B = self.preluB3(
            self.convB3(torch.cat(
                (F.interpolate(imageB, [H, W], mode='nearest'), self.preluB2(self.convB2(ca_B)) + imageB_sideout), 1)))

        return ca_A, ca_B


class text_preprocess(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(text_preprocess, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class attention_mapping(nn.Module):
    def __init__(self, in_dim=256):
        super(attention_mapping, self).__init__()
        self.q_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.q_p_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.q_c_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.dk = in_dim
        self.layer_norm = nn.LayerNorm(in_dim, elementwise_affine=False)

    def forward(self, x, embd=None):
        B, C, H, W = x.size()
        q = self.q_conv(x)
        v = self.v_conv(x)
        q_p = self.q_p_conv(x)
        q_c = self.q_c_conv(x)

        q_flat = q.view(B, C, -1).permute(0, 2, 1)
        q_p_flat = q_p.view(B, C, -1).permute(0, 2, 1)
        q_c_flat = q_c.view(B, C, -1).permute(0, 2, 1)
        v_flat = v.view(B, C, -1).permute(0, 2, 1)

        if embd is not None:
            k = embd.mean(dim=1).unsqueeze(1).unsqueeze(2)
        else:
            k = q.mean(dim=[2, 3], keepdim=True)

        score = torch.bmm(q_flat, k.view(B, C, 1)) / math.sqrt(self.dk)
        score_p = torch.bmm(q_p_flat, k.view(B, C, 1)) / math.sqrt(self.dk)
        score_c = torch.bmm(q_c_flat, k.view(B, C, 1)) / math.sqrt(self.dk)

        attention = F.softmax(score + score_p + score_c, dim=1)

        out_flat = torch.bmm(attention.transpose(1, 2), v_flat)
        out = out_flat.expand(-1, H * W, -1).permute(0, 2, 1).view(B, C, H, W)
        out = out.permute(0, 2, 3, 1)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)
        return out


class att_module(nn.Module):
    def __init__(self, in_channels=1, embd_dim=256, epsilon=0.9):
        super(att_module, self).__init__()
        self.pixelnorm = PixelNorm()
        self.in_channels = in_channels
        self.embd_dim = embd_dim
        self.epsilon = epsilon

        self.input_conv = nn.Conv2d(in_channels, embd_dim, kernel_size=3, stride=1, padding=1)
        self.attention_mapping = attention_mapping(in_dim=embd_dim)
        self.out_conv = nn.Conv2d(embd_dim, in_channels, kernel_size=1)

        # 使用Tanh代替Sigmoid
        self.tanh = nn.Tanh()

        # 添加局部响应归一化层
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

    def forward(self, x, embd=None):
        # residual = x.clone()  # 保存原始输入作为残差连接
        # x = self.input_conv(x)

        # x = self.pixelnorm(x)

        # t = self.attention_mapping(x, embd)

        # h_att = (1 - self.epsilon) * t

        # h = self.out_conv(h_att)

        # # 使用Tanh函数增加对比度
        # h = self.tanh(h)

        # # 应用局部响应归一化增强局部对比度
        # h = self.lrn(h)

        # # 强化残差连接 R + S
        # h = self.epsilon * residual + (1 - self.epsilon) * h

        #1 without S
        residual = x.clone()  # 保存原始输入作为残差连接
        x = self.input_conv(x)

        x = self.pixelnorm(x)

        t = self.attention_mapping(x, embd)

        h_att = (1 - self.epsilon) * t

        h = self.out_conv(h_att)

        # 使用Tanh函数增加对比度
        h = self.tanh(h)

        # 应用局部响应归一化增强局部对比度
        h = self.lrn(h)

        # 强化残差连接 R + S
        h = self.epsilon * residual 


        return h


class Net(nn.Module):
    def __init__(
            self,
            mid_channel=32,
            decoder_num_heads=8,
            ffn_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            out_channel=1,
            hidden_dim=256,
            image2text_dim=32,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        self.text_process = text_preprocess(768, hidden_dim)
        self.restormerca1 = restormer_cablock(hidden_dim=hidden_dim, image2text_dim=image2text_dim)
        self.restormerca2 = restormer_cablock(input_channel=mid_channel, hidden_dim=hidden_dim,
                                              image2text_dim=image2text_dim)
        self.restormerca3 = restormer_cablock(input_channel=mid_channel, hidden_dim=hidden_dim,
                                              image2text_dim=image2text_dim)
        self.restormer1 = Restormer(2 * mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer2 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer3 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.conv1 = nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.att_mapper = att_module(in_channels=1, embd_dim=256)
        self.softmax = nn.Sigmoid()

    def forward(self, imageA, imageB, text):
        text = self.text_process(text)
        '''
            imageA：    torch.Size([2, 1, 256, 256])  torch.Size([2, 1, 256, 256])
            featureA：  torch.Size([2, 32, 256, 256])
        '''
        alinged_imageA = self.att_mapper(imageA, text)
        alinged_imageB = self.att_mapper(imageB, text)
        featureA, featureB = self.restormerca1(alinged_imageA, alinged_imageB, text)
        # featureA, featureB = self.restormerca1(imageA, imageB, text)
        featureA, featureB = self.restormerca2(featureA, featureB, text)
        featureA, featureB = self.restormerca3(featureA, featureB, text)
        featureA, featureB = self.restormerca3(featureA, featureB, text)
        # 图像特征与文本特征进行语义注入

        fusionfeature = torch.cat((featureA, featureB), 1)
        fusionfeature = self.restormer1(fusionfeature)
        fusionfeature = self.conv1(fusionfeature)
        fusionfeature = self.restormer2(fusionfeature)
        fusionfeature = self.restormer3(fusionfeature)
        fusionfeature = self.restormer3(fusionfeature)
        fusionfeature = self.conv2(fusionfeature)
        fusionfeature = self.softmax(fusionfeature)
        return fusionfeature