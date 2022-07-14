#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:28:40 2022

@author: surajb
"""
import math

def output_dim_conv(h_in,w_in,kernel,stride,padding=[1,0]):
    h_out = math.floor(((h_in + 2*padding[0] -kernel[0])/stride[0])+1)
    w_out = math.floor(((w_in + 2*padding[1] -kernel[1])/stride[1])+1)
    return h_out,w_out


def output_dim_pool(h_in,w_in,kernel,stride,padding=[0,0]):
    h_out = math.floor(((h_in-kernel[0])/stride[0])+1)
    w_out = math.floor(((w_in-kernel[1])/stride[1])+1)
    return h_out,w_out



def output_dim_trans_conv(h_in,w_in,kernel,op_padding,stride,padding=[1,1]):
    h_out = math.floor(stride[0]*(h_in - 1) + kernel[0] - 2*padding[0] + op_padding[0])
    w_out = math.floor(stride[1]*(w_in - 1) + kernel[1] - 2*padding[1]+ op_padding[1])
    
   
    return h_out,w_out


h_img = 2
w_img = 3

kernel_conv =[3,3]
stride_conv = [1,1]

kernel_pool =[3,3]
stride_pool = [2,2]

kernel_trans_conv =[3,3]
stride_trans_conv = [2,2]
op_padding_trans_conv = [1,0]


h_out_conv,w_out_conv = output_dim_conv(h_img,w_img,kernel=kernel_conv,stride=stride_conv)

h_out_pool,w_out_pool = output_dim_pool(h_img,w_img,kernel=kernel_pool,stride=stride_pool)


h_out_trans_conv,w_out_trans_conv = output_dim_trans_conv(h_img,w_img,kernel=kernel_trans_conv,op_padding=op_padding_trans_conv,stride=stride_trans_conv)

print('input: ',h_img,',',w_img,';','output_conv: ',h_out_conv,',',w_out_conv)
print('input: ',h_img,',',w_img,';','output_pool: ',h_out_pool,',',w_out_pool)
print('input: ',h_img,',',w_img,';','output_trans_conv: ',h_out_trans_conv,',',w_out_trans_conv)