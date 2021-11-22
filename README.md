## Unsupervised Holistic Image Generation from Key Local Patches
paper link: [https://arxiv.org/pdf/1703.10730.pdf]

參考這篇論文並利用數字辨識簡單實現。

先在同個資料夾中建立 model_checkpoint & Result
分別存放 model 權重以及 生成結果

---
### model.py 
model 部分在 convlution layer 的 output_channel 有稍微修正過，架構上基本一致
在 discriminator 的部分因是採用 PatchGAN 的架構

---
### losses.py 
原則上採用 paper 上面的 loss 設定，本人在做的時候 training 比較不容易，所以 D 的架構有參考 LSGAN 以及 PatchGAN 的架構

---
### prepare_data.py 
原 paper 是以 celebA 當作人臉的訓練
這裡以手寫數字的 mnist 做簡單訓練
這裡的實驗在 mask 選法上，目前是以隨機的方式選擇，並沒有做太複雜的運算
