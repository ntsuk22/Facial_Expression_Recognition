# FER-2013 — სახის ემოციების აღიარება (Facial Expression Recognition)

**Kaggle Competition:** [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

**WandB Experiments:** [https://wandb.ai/ntsuk22-free-university-of-tbilisi-/fer2013-experiments](https://wandb.ai/ntsuk22-free-university-of-tbilisi-/fer2013-experiments?nw=nwuserntsuk22)

---

## პროექტის მიმოხილვა

ეს პროექტი წარმოადგენს FER-2013 dataset-ზე ნეირონული ქსელების იტერაციული განვითარების პრაქტიკულ გამოცდილებას. მიზანი იყო არა მხოლოდ მაღალი accuracy-ის მიღწევა, არამედ სხვადასხვა არქიტექტურებისა და hyperparameter-ების სისტემატური შედარება, overfitting და underfitting-ის პატერნების იდენტიფიცირება და მათი ანალიზი.

სამი არქიტექტურა გავიარე კრებადობით:

```
SimpleCNN (~50K params)  →  DeeperCNN (~800K params)  →  ResNetCNN (~1.2M params)
      ~38% val acc               ~64% val acc                  ~66.7% val acc
    underfitting              plateau / balanced              residual + SE attention
```

---

## Dataset

**FER-2013** — 48×48 grayscale სურათები, 7 ემოციის კლასი

| Split | სურათების რაოდენობა |
|-------|-------------------|
| Train | 28,709 |
| Validation (PublicTest) | 3,589 |

**კლასები:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

კლასების განაწილება **არათანაბარია** — Happy კლასი მნიშვნელოვნად ჭარბობს, Disgust კი ყველაზე ნაკლებია. ეს ართულებს სწავლას და ითხოვს ყურადღებას augmentation სტრატეგიის შერჩებისას.

---

## არქიტექტურა 1 — SimpleCNN

### სტრუქტურა

```
Input (1×48×48)
  → Conv2d(1→32, 3×3) + BatchNorm + ReLU + MaxPool
  → Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool
  → AdaptiveAvgPool → Dropout → Linear(64→128) → ReLU → Linear(128→7)
```

~50K parameter. გამოიყენება baseline-ად — მიზანი იყო hyperparameter-ების გავლენის შესწავლა არქიტექტურის გართულების გარეშე.

### Experiment 1 — Baseline

**კონფიგურაცია:** lr=1e-3, dropout=0.3, weight_decay=1e-4, CosineAnnealingLR, 30 epoch

**შედეგი:** val acc ~38%, train acc ~34%, gap ~0.02 → **Underfitting**

Adam optimizer-ისთვის lr=1e-3 სტანდარტული საწყისი წერტილია. dropout=0.3 მსუბუქი regularization-ია, რომ underfitting არ გამოიწვიოს. CosineAnnealingLR training-ის ბოლოს lr-ს 0-მდე ამცირებს, რაც fine-tuning ეფექტს ქმნის. თუმცა train და val accuracy-ს შორის gap თითქმის ნულოვანი აღმოჩნდა — მოდელი training set-ის სწავლასაც ვერ ახერხებდა. ეს capacity-ის ნაკლებობის კლასიკური ნიშანია.

### Experiment 2 — Lower LR + Higher Dropout

**ცვლილებები:** lr: 1e-3 → **1e-4**, dropout: 0.3 → **0.5**

**შედეგი:** val acc ~35%, train acc ~30%, gap ~0.02 → **გაღრმავებული Underfitting**

exp1-ში loss ნელა იკლებდა, ამიტომ ვიფიქრე, რომ lr შემცირება უფრო ზუსტ gradient descent-ს მომცემდა. თუმცა lr=1e-4 კონვერგენციას კიდევ უფრო შეანელა — 30 epoch-ში მოდელი ნაკლებ სწავლობდა. dropout=0.5-ის გაზრდამ კი სწავლა ამ მოდელისთვის შეუძლებლად გახადა. ექსპერიმენტმა დაადასტურა: **პრობლემა overfitting არ არის, regularization-ის გაძლიერება ვერ გამოასწორებს.**

### Experiment 3 — No Scheduler + High Weight Decay

**ცვლილებები:** lr დაბრუნდა 1e-3-ზე, dropout: 0.5 → **0.3**, weight_decay: 1e-4 → **1e-2**, scheduler: **გამოირთო**

**შედეგი:** val acc ~34%, train acc ~29%, gap ~0.05 → **ყველაზე ძლიერი Underfitting**

scheduler გამოვრთე, რომ მენახა რამდენს მაძლევს CosineAnnealingLR. სტაბილური lr=1e-3 training loss-ს უფრო სტაბილურად ამცირებდა, მაგრამ weight_decay=1e-2 ამ ზომის მოდელისთვის ძალიან მკაცრი აღმოჩნდა — weight-ები იმდენად პატარა დარჩა, რომ მოდელი feature-ების სწავლას ვეღარ ახერხებდა. ეს ყველაზე ცუდი შედეგი გახდა.

### SimpleCNN — მთავარი დასკვნა

ყველა სამი ექსპერიმენტი underfitting-ს ავლენდა. ~50K parameter ამ dataset-ისთვის საკმარისი არ არის. **underfitting-ის გამოსწორება regularization-ის შეცვლით შეუძლებელია** — გამოსავალი მხოლოდ არქიტექტურის გაფართოებაა.

---

## არქიტექტურა 2 — DeeperCNN

### სტრუქტურა

```
Input (1×48×48)
  → ConvBlock(1→32):   Conv+BN+ReLU → Conv+BN+ReLU → MaxPool → Dropout2d  →  32×12×12
  → ConvBlock(32→64):  Conv+BN+ReLU → Conv+BN+ReLU → MaxPool → Dropout2d  →  64×6×6
  → ConvBlock(64→128): Conv+BN+ReLU → Conv+BN+ReLU → MaxPool → Dropout2d  →  128×6×6
  → Flatten → Linear(4608→256) → ReLU → Dropout → Linear(256→128) → ReLU → Linear(128→7)
```

~800K parameter. SimpleCNN-ის underfitting-ის პასუხი.

**ძირითადი სხვაობები SimpleCNN-თან:**
- მე-3 conv block დაემატა (64→128 channels)
- ყოველ block-ში 2 conv layer (VGG-style double conv) — ორი 3×3 conv იძლევა 5×5-ის ეკვივალენტური receptive field-ს ნაკლები parameter-ებით და დამატებითი nonlinearity-ით
- Flatten ნაცვლად GlobalAvgPool-ისა — 6×6 feature map-ების სრული სტრუქტურა ინახება, რაც ემოციების ლოკალური ნიშნებისთვის (თვალი, პირი, წარბი) მნიშვნელოვანია
- Dropout2d conv block-ებშიც, არა მხოლოდ FC layer-ებში

### Experiment 1 — Baseline

**კონფიგურაცია:** lr=1e-3, weight_decay=1e-4, conv_dropout=0.25, fc_dropout=0.5, ReduceLROnPlateau (factor=0.5, patience=5), ძლიერი augmentation (flip + rotation(15) + RandomResizedCrop + ColorJitter)

**შედეგი:** val acc 64.08%, train acc 60.39%, gap -0.018 → **Reasonable fit (over-augmented)**

ReduceLROnPlateau ავარჩიე, რადგან ვფიქრობდი, რომ ადაპტური scheduler უფრო დიდ მოდელზე გამართლებული იქნებოდა — lr-ი ნახევრდება მხოლოდ მაშინ, როდესაც val_loss 5 epoch-ით არ გაუმჯობესდება. ძლიერი augmentation გამოვიყენე robustness-ისთვის. თუმცაღა, ReduceLROnPlateau-მ lr საერთოდ არ შეამცირა, რადგან val_loss მთელი 50 epoch-ის განმავლობაში სტაბილურად იკლებდა. ყველაზე მნიშვნელოვანი: **gap უარყოფითი გახდა (-0.018)** — train acc < val acc. ეს over-augmentation-ის ნიშანია: training სურათები validation სურათებზე გაცილებით გართულდა. SimpleCNN-თან შედარებით (38%) val accuracy 64%-მდე გაიზარდა.

### Experiment 2 — Less Augmentation + More Epochs

**ცვლილებები:** epochs: 50 → **70**, rotation: 15° → **10°**, scheduler: ReduceLROnPlateau → **CosineAnnealingLR**

**შედეგი:** val acc 64.28%, train acc 62.72%, gap -0.012 → **Reasonable fit**

exp1-ში val_loss 50 epoch-ზე ჯერ კიდევ იკლებდა — plateau ჯერ არ მიღწეულა, ამიტომ epoch-ების გაზრდა გამართლებული იყო. CosineAnnealingLR ავარჩიე, რადგან ReduceLROnPlateau-მ lr საერთოდ არ შეცვალა. CosineAnnealingLR გარანტირებულად ამცირებს lr-ს — training-ის ბოლო ფაზაში fine-tuning ეფექტი. rotation-ის 15°→10°-ზე შემცირება exp1-ის უარყოფითი gap-ის პასუხი იყო. gap შემცირდა -0.018-დან -0.012-მდე, val acc ოდნავ გაიზარდა. ეს სწორი მიმართულება იყო.

### Experiment 3 — No Augmentation

**ცვლილებები:** augmentation: **მხოლოდ flip** (crop და jitter მოიხსნა), epochs: 70 → **60**

**შედეგი:** val acc 63.72%, train acc 78.97%, gap +0.156 → **Overfitting**

ეს ექსპერიმენტი დიაგნოსტიკური მიზნით ჩავატარე: მინდოდა მენახა, შეძლებდა თუ არა მოდელი training set-ის სწავლას. augmentation-ის მოხსნამ train accuracy 63%→79%-მდე გაზარდა — **მოდელს capacity აქვს.** მაგრამ val accuracy შემცირდა (63.9%→63.4%), gap-ი კი +0.156-მდე გაიზარდა. ეს კლასიკური overfitting-ია: მოდელი training სურათებს „იმახსოვრებდა" generalization-ის გარეშე. ეს დასკვნა მნიშვნელოვანია — underfitting-ი აღარ მაქვს, regularization სჭირდება.

### Experiment 4 — Stronger Regularization

**ცვლილებები:** weight_decay: 1e-4 → **1e-3** (10x), conv_dropout: 0.25 → **0.40**, augmentation: **flip + rotation(10)** დაბრუნდა

**შედეგი:** val acc 63.28%, train acc 62.72%, gap -0.002 → **Balanced / Plateau**

exp3-ის gap +0.156 პირდაპირ მიუთითებდა regularization-ის საჭიროებაზე. weight_decay 10-ჯერ გავზარდე — L2 regularization weight-ებს პატარა ამპლიტუდასთან ახლოს ინახავს. conv_dropout 0.25→0.40 feature map-ების დონეზე regularization-ს ამძაფრებს, რაც exp3-მა მაჩვენა, რომ overfitting conv layer-ებშიც ხდება. augmentation exp2-ის ოპტიმალურ დონეზე დავაბრუნე. regularization-მა overfitting მოხსნა (gap: +0.156 → -0.002), მაგრამ val acc არ გაიზარდა. **~63–64% DeeperCNN-ის ceiling-ია** — ნებისმიერი regularization ამ ფარგლებში ინახება.

### DeeperCNN — მთავარი დასკვნა

DeeperCNN-მა SimpleCNN-ის ceiling გარღვია (38%→64%). ოთხი ექსპერიმენტი ოთხ სხვადასხვა პრობლემას წააწყდა: over-augmentation, late plateau, overfitting, balanced plateau. **~64% ამ არქიტექტურის ზედა ზღვარია** — შემდეგი ნაბიჯი არქიტექტურის შეცვლაა.

---

## არქიტექტურა 3 — ResNetCNN

### სტრუქტურა

```
Input (1×48×48)
  Stem:   Conv(1→32, 3×3) + BN + ReLU                        → 32×48×48
  layer1: ResBlock(32→64,  stride=2) + SE                     → 64×24×24
          ResBlock(64→64,  stride=1) + SE                     → 64×24×24
  layer2: ResBlock(64→128, stride=2) + SE                     → 128×12×12
          ResBlock(128→128,stride=1) + SE                     → 128×12×12
  layer3: ResBlock(128→256,stride=2) + SE                     → 256×6×6
  GlobalAvgPool → Dropout(fc_dropout) → Linear(256→128) → ReLU → Linear(128→7)
```

~1.2M parameter.

**ძირითადი კომპონენტები:**

**Residual (Skip) Connections** — DeeperCNN-ში gradient-ი 3 plain conv block-ზე გადასვლისას სუსტდებოდა (vanishing gradient). skip connection-ი gradient-ს პირდაპირ გაატარებს — მოდელი სწავლობს identity-ს residual-ს, არა სრულ ტრანსფორმაციას ნულიდან.

```
x → [Conv-BN-ReLU → Conv-BN] → + x → ReLU
                                ↑
                           skip connection
```

**Squeeze-and-Excitation (SE) Block** — სწავლობს per-channel importance weight-ებს. FER-2013-ისთვის კრიტიკულია: Happy ემოციისთვის „ღიმილის" channel-ი მაღალ წონას იღებს, Angry-სთვის — სხვა channel-ები.

```
feature maps → GlobalAvgPool → FC(C→C/16) → ReLU → FC(C/16→C) → Sigmoid → scale
```

**DeeperCNN-ში DeeperCNN-ის ceiling-ის 3 მიზეზი:**
1. Vanishing gradient — 3 plain conv block-ში gradient-ი სუსტდება
2. Feature reuse-ის არარსებობა — ყოველი block ნულიდან სწავლობს
3. Channel attention-ის არარსებობა — ყველა feature map თანაბარ წონას იღებს

### Experiment 1 — Baseline

**კონფიგურაცია:** lr=1e-3, weight_decay=1e-4, fc_dropout=0.5, se_reduction=16, CosineAnnealingLR, 60 epoch, flip + rotation(10)

**შედეგი:** val acc 66.98%, train acc 94.70%, gap +0.288 → **Severe Overfitting**

DeeperCNN exp2-ის საუკეთესო augmentation დონე და CosineAnnealingLR გადმოვიტანე. fc_dropout=0.5 სტანდარტული საწყისი წერტილია FC head-ისთვის. მაგრამ val accuracy-მ (65.9%) DeeperCNN-ის ceiling-ი გარღვია — **არქიტექტურა მუშაობს.** ამავდროულად, train accuracy 94.7%-მდე მიაღწია. residual connections-ი + SE attention-ი მოდელს მნიშვნელოვნად მეტ representational power-ს აძლევს — ეს capacity მინიმალური regularization-ის პირობებში training set-ის memorization-ისთვის გამოიყენება. gap 0.288 ყველაზე დიდია ყველა ექსპერიმენტში.

### Experiment 2 — Stronger Regularization

**ცვლილებები:** weight_decay: 1e-4 → **1e-3** (10x), fc_dropout: 0.5 → **0.6**, augmentation: **flip + rot(15) + RandomResizedCrop + ColorJitter**

**შედეგი:** val acc 66.37%, train acc 70.93%, gap +0.046 → **Reasonable fit**

exp1-ის gap 0.288 პირდაპირ მიუთითებდა სამ საჭირო ცვლილებაზე. weight_decay 10-ჯერ გავზარდე — L2 regularization weight-ებს ამცირებს და memorization-ს ართულებს. fc_dropout 0.5→0.6 FC head-ში overfitting-ს ამცირებს. RandomResizedCrop + ColorJitter training სურათებს ყოველ epoch-ში სხვადასხვაგვარ ხედვას აჩვენებს — memorization-ი ამ პირობებში გაცილებით უფრო რთულია. gap 0.288-დან 0.046-მდე ჩამოვიდა, train accuracy 94.7%→70.9%. val accuracy (~66%) სტაბილური დარჩა — ეს ადასტურებს, რომ exp1-ის val accuracy ნამდვილი იყო. CosineAnnealingLR-ის ბოლო epoch-ებში gap ოდნავ იზრდება — lr→0-მდე კლება late-stage overfitting-ს იწვევს.

### Experiment 3 — OneCycleLR + Label Smoothing

**ცვლილებები:** scheduler: CosineAnnealingLR → **OneCycleLR**, loss: CrossEntropyLoss → **CrossEntropyLoss(label_smoothing=0.1)**, epochs: 60 → **100**, batch_size: 64 → **128**

**შედეგი:** val acc 66.98%, train acc 72.60%, gap +0.060 → **Reasonable fit ✓ საუკეთესო**

exp2-ში CosineAnnealingLR lr-ს monotonically ამცირებდა — ბოლო epoch-ებში gap იზრდებოდა. OneCycleLR warmup-ით იწყება (lr თანდათან იზრდება), peak-ს აღწევს, შემდეგ smoothly კლებულობს. warmup phase-ი ადრეულ training-ში instability-ს თავიდან არიდებს, ხოლო smooth decay late-stage overfitting-ს ამცირებს. label smoothing (0.1) ნაცვლად correct class-ისთვის probability 1.0-ის, 0.9-ს target-ად იყენებს — მოდელი ნაკლებ overconfident ხდება. ეს train loss-ის შედარებაშიც ჩანს: exp2 — 0.795, exp3 — 1.034. epoch-ების 60→100-მდე გაზრდა OneCycleLR-ს warmup და decay phase-ებისთვის საჭირო სივრცეს აძლევს. batch_size 64→128 gradient-ების estimate-ებს უფრო სტაბილურს ხდის. საბოლოო შედეგი: val acc exp1-ის იდენტური (66.98%), მაგრამ gap-ი კონტროლირებულია (0.060 vs 0.288).

### ResNetCNN — მთავარი დასკვნა

ResNetCNN-მა DeeperCNN-ის ceiling გარღვია (64%→66.7%). skip connections-ი vanishing gradient-ს გადაჭრა, SE attention-ი per-channel importance-ს სწავლობს. **late-stage overfitting პერსისტენტული პატერნია** — ყველა სამ ექსპერიმენტში training-ის ბოლოს gap იზრდება lr→0-სთან ერთად.

---

## ჯამური შედარება

| არქიტექტურა | Best Val Acc | Gap | Parameter-ები | დიაგნოზი |
|-------------|:-----------:|:---:|:-------------:|---------|
| SimpleCNN exp1 | ~38% | ~0.02 | 50K | Underfitting |
| SimpleCNN exp2 | ~35% | ~0.02 | 50K | Underfitting გაძლიერდა |
| SimpleCNN exp3 | ~34% | ~0.05 | 50K | ყველაზე ძლიერი Underfitting |
| DeeperCNN exp1 | 64.08% | -0.018 | 800K | Over-augmentation |
| DeeperCNN exp2 | 64.28% | -0.012 | 800K | Reasonable fit ✓ |
| DeeperCNN exp3 | 63.72% | +0.156 | 800K | Overfitting |
| DeeperCNN exp4 | 63.28% | -0.002 | 800K | Balanced Plateau |
| ResNetCNN exp1 | 66.98% | +0.288 | 1.2M | Severe Overfitting |
| ResNetCNN exp2 | 66.37% | +0.046 | 1.2M | Reasonable fit |
| **ResNetCNN exp3** | **66.98%** | **+0.060** | **1.2M** | **Reasonable fit ✓ საუკეთესო** |

---

## Overfitting / Underfitting — ჯამური ანალიზი

### Underfitting (SimpleCNN ყველა ექსპერიმენტი)

**ნიშნები:** train acc ≈ val acc ≈ დაბალი (~30–38%), gap ≈ 0

**მიზეზი:** ~50K parameter FER-2013-ის სირთულეს ვერ პასუხობს. 2 conv block ვერ სწავლობს 7 ემოციის კლასს 48×48 სურათებში.

**რას ვცადე და რატომ არ გამოვიდა:** lr-ის შემცირება კონვერგენციას ანელებდა, dropout-ის გაზრდა სწავლას კიდევ უფრო ართულებდა, weight_decay-ის გაზრდა weight-ებს „ჩახრჩობდა". **underfitting-ის გამოსავალი მხოლოდ capacity-ის გაზრდაა.**

### Overfitting

**DeeperCNN exp3 (gap +0.156):** augmentation-ის მოხსნამ memorization-ი გამოიწვია. train acc 63%→79%, val acc კი შემცირდა. **გამოსავალი:** augmentation-ის დაბრუნება + regularization-ის გაძლიერება.

**ResNetCNN exp1 (gap +0.288):** skip connections + SE → მაღალი representational power → training set memorization მინიმალური regularization-ით. train acc 94.7%. **გამოსავალი:** weight_decay ↑, fc_dropout ↑, stronger augmentation.

### Negative Gap — Over-augmentation (DeeperCNN exp1, exp2)

Train acc < val acc — training სურათები validation-ზე გაცილებით რთული ხდება RandomResizedCrop + ColorJitter + rotation(15)-ის კომბინაციით. **გამოსავალი:** augmentation-ის შემსუბუქება (rotation შემცირება, crop/jitter მოხსნა).

---

## WandB Tracking

ყველა ექსპერიმენტი დალოგილია WandB-ზე:

**[https://wandb.ai/ntsuk22-free-university-of-tbilisi-/fer2013-experiments](https://wandb.ai/ntsuk22-free-university-of-tbilisi-/fer2013-experiments?nw=nwuserntsuk22)**

ლოგირებული მეტრიკები:
- `train/loss`, `train/accuracy` — ყოველი epoch
- `val/loss`, `val/accuracy` — ყოველი epoch
- `lr` — learning rate history
- hyperparameter-ები — run config-ში

სტრუქტურა: სხვადასხვა არქიტექტურა — სხვადასხვა run. run-ების სახელები: `simple_cnn_exp1`, `deeper_cnn_exp2_less_aug`, `resnet_cnn_exp3_warmup_cosine` და სხვ.

---

## Repository სტრუქტურა

```
fer2013/
├── README.md
├── models/
│   ├── simple_cnn.py
│   ├── deeper_cnn.py
│   └── resnet_cnn.py
├── train.py              ← სასწავლო script (WandB ინტეგრაციით)
├── data.py               ← FER-2013 dataloader + augmentation
└── notebooks/
    ├── SimpleCNN_experiments.ipynb
    ├── DeeperCNN_experiments.ipynb
    └── ResNetCNN_experiments.ipynb
```

---

## შემდეგი ნაბიჯები

ResNetCNN-მა 66.7% val accuracy მიაღწია. train (72.6%) და val (66.7%) შორის gap-ი ჯერ კიდევ არსებობს. შემდგომი გაუმჯობესებისთვის:

- **Transfer learning** — ImageNet-ზე pre-trained model-ების ადაპტაცია grayscale-ზე
- **CBAM attention** — channel + spatial attention კომბინაცია (SE-ს განვითარება)
- **Class-weighted loss** — Disgust კლასის imbalance-ის კომპენსაცია
- **Early stopping** — late-stage overfitting-ის პრევენცია
