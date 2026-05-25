//一共2722个数据:
//  actor_net.0.weight | 包含 512 个 float   
//  actor_net.0.bias | 包含 64 个 float
//  actor_net.2.weight | 包含 2048 个 float   
//  actor_net.2.bias | 包含 32 个 float
//  actor_net.4.weight | 包含 64 个 float
//  actor_net.4.bias | 包含 2 个 float
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<weights_data.h>


// in_n x hidden_n * hidden_n x out_n + out_n x 1 = in_n x out_n
// X * W (actually W is out x hidden shape in physical)
void linear(const float* src,const float* w,const float* b,
              int in_n,int hidden_n,int out_n,float* dst)
{
    // dst[i][j]= src[i][*] dot w[*][j]
    for(int i=0;i<in_n;i++)
        for(int j=0;j<out_n;j++)
        {
            //计算xw+b中+b, 同时也解决了脏数据初始化问题;
            float sum=b[j];
            for(int k=0;k<hidden_n;k++)
            {
                sum+=src[i*hidden_n+k]*w[j*hidden_n+k];
            }
            dst[i*out_n+j]=sum;
        }
}

void tahn(float* src,int size)
{
    for(int i=0;i<size;i++)
        src[i]=tanhf(src[i]);
}

int main(){

    const float *w1,*b1,*w2,*b2,*w3,*b3; //三层
    w1=&weights_pool[0];
    b1=&weights_pool[512];
    w2=&weights_pool[512+64];
    b2=&weights_pool[512+64+2048];
    w3=&weights_pool[512+64+2048+32];
    b3=&weights_pool[512+64+2048+32+64];

    // forward: 1x8 * 8x64 +64 -> 1x64 * 64x32 +32 -> 1x32 * 32x2 +2 -> 1x2
    //中间变量可以一种复用(逻辑简单不涉及异步,share,有严格同步逻辑)
    float buffer1[64];
    float buffer2[64];
    //样本调试用;
    float sample[8]={0.1382, 0.0692, 0.5468, 0.6220, 0.8029, 0.7487, 0.2163, 0.6195};
    
    for(int i=0;i<8;i++)
    {
        sample[i]=(sample[i]-mean[i])/(std[i]+1e-8f);
    }
    
    linear(sample,w1,b1,1,8,64,buffer1);   // layer1 1x8,8x64
    tahn(buffer1,64);     
    linear(buffer1,w2,b2,1,64,32,buffer2);   //layer2 1x64,64x32
    tahn(buffer2,32);
    linear(buffer2,w3,b3,1,32,2,buffer1);    //layer3 1x32,32x2
    tahn(buffer1,2);
    
    float u_L=buffer1[0];
    float u_R=buffer1[1];
    
    printf("u_L, u_R is : %f %f",u_L,u_R);

// tensor([[0.1382, 0.0692, 0.5468, 0.6220, 0.8029, 0.7487, 0.2163, 0.6195]])
// tensor([[-0.2138, -0.2141]], grad_fn=<TanhBackward0>)

    

}

