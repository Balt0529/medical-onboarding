- 必要最低限のことを適当に書きました
- みんな頑張ろう！

# Dataset 
### データセット発行元の紹介
Alzheimer’s Disease Neuroimaging Initiative(ADNI)データセット: 軽度認知障害やアルツハイマーの研究を目的として収集された脳MR画像のデータセットです。

公式サイト: https://adni.loni.usc.edu/

### クラスラベル
今回使用するクラスラベルを下記に示します。
- Alzheimer’s Disease(AD): アルツハイマー病患者
- Coginively Normal(CN): 健常者
- Mild Cognitive Impairment(MCI): 軽度認知障害患者

# 前処理
提供されているADNIデータセットには、macky先輩が作成したOpen-MAPT1手法を用いて前処理を施しています。皆が見るデータはこの手法を用いて非脳組織が除去されたものになっていると思います。

論文URL: https://www.medrxiv.org/content/10.1101/2024.01.18.24301494v1

前処理前の画像を示します。

<p align="center">
  <a href="./前処理前の画像.png">
    <img width="300px" src="./前処理前の画像.png" />
  </a>
</p>

前処理後の画像を示します。上から冠状面、横断面、矢状面です。
<p align="center">
  <a href="./冠状面.png">
    <img width="300px" src="./冠状面.png" />  
  </a>
</p>

<p align="center">
  <a href="./横断面.png">
    <img width="300px" src="./横断面.png" />  
  </a>
</p>

<p align="center">
  <a href="./矢状面.png">
    <img width="300px" src="./矢状面.png" /> 
  </a>
</p>


# やってほしいこと
- 上に載せたような冠状面、横断面、矢状面を実際に自分でコードを表示させてみよう
- ADとCNの2クラス分類を行ってみよう。精度が80%ぐらい出るはず

### 発展
- tsneやUMAPを用いて、データを二次元に可視化してみよう
