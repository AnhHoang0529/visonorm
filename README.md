# 📦 ViSoNorm Toolkit — Vietnamese Text Normalization & Processing

**ViSoNorm** là một toolkit chuyên biệt dành cho **chuẩn hóa và xử lý văn bản tiếng Việt**, được thiết kế tối ưu cho môi trường **NLP** và dễ dàng cài đặt qua **PyPI**. Các tài nguyên (datasets, models) được lưu trữ và quản lý trực tiếp trên **Hugging Face Hub** và **GitHub Releases**.

[![PyPI version](https://badge.fury.io/py/visonorm.svg)](https://badge.fury.io/py/visonorm)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Tính năng chính

### 1. 🔧 **BasicNormalizer** — Chuẩn hóa văn bản cơ bản

* **Case folding**: chuyển toàn bộ văn bản về lowercase/uppercase/capitalize.
* **Tone normalization**: chuẩn hóa dấu thanh tiếng Việt.
* **Basic preprocessing**: loại bỏ khoảng trắng thừa, ký tự đặc biệt, định dạng câu.

### 2. 😀 **EmojiHandler** — Xử lý emoji

* **Detect emojis**: phát hiện emoji trong văn bản.
* **Split emoji text**: tách emoji ra khỏi câu.
* **Remove emojis**: loại bỏ toàn bộ emoji.

### 3. ✏️ **Lexical Normalization** — Chuẩn hóa văn bản mạng xã hội

* **ViSoLexNormalizer**: Chuẩn hóa văn bản sử dụng mô hình deep learning từ HuggingFace.
* **NswDetector**: Phát hiện từ phi chuẩn (non-standard words).
* **detect_nsw()**: Hàm tiện ích để phát hiện NSW.
* **normalize_sentence()**: Hàm tiện ích để chuẩn hóa câu.

### 4. 📊 **Resource Management** — Quản lý dữ liệu

* `list_datasets()` — Liệt kê datasets có sẵn.
* `load_dataset()` — Tải dataset từ GitHub Releases.
* `get_dataset_info()` — Xem thông tin chi tiết dataset.

### 5. 🧠 **Task Models** — Mô hình xử lý tác vụ

* **SpamReviewDetection** — Phát hiện spam.
* **HateSpeechDetection** — Phát hiện hate speech.
* **HateSpeechSpanDetection** — Phát hiện span của hate speech.
* **EmotionRecognition** — Nhận diện cảm xúc.
* **AspectSentimentAnalysis** — Phân tích sentiment theo từng khía cạnh.

---

## 📥 Cài đặt

### Cài đặt từ PyPI (Khuyến nghị)

```bash
pip install visonorm
```

### Requirements

- Python >= 3.10
- PyTorch >= 1.10.0
- Transformers >= 4.0.0
- scikit-learn >= 0.24.0
- pandas >= 1.3.0

---

## 📚 Hướng dẫn sử dụng

### 1. 🔧 BasicNormalizer — Chuẩn hóa văn bản cơ bản

```python
from visonorm import BasicNormalizer

# Khởi tạo BasicNormalizer
normalizer = BasicNormalizer()

# Ví dụ văn bản
text = "Hôm nay tôi rất VUI 😊 và HẠNH PHÚC 🎉!"

# Case folding
print(normalizer.case_folding(text, mode='lower'))
# Output: hôm nay tôi rất vui 😊 và hạnh phúc 🎉!

print(normalizer.case_folding(text, mode='upper'))
# Output: HÔM NAY TÔI RẤT VUI 😊 VÀ HẠNH PHÚC 🎉!

print(normalizer.case_folding(text, mode='capitalize'))
# Output: Hôm Nay Tôi Rất Vui 😊 Và Hạnh Phúc 🎉!

# Tone normalization
text2 = "Bận xong rồi. Xoã đi :)"
print(normalizer.tone_normalization(text2))
# Output: Bận xong rồi. Xõa đi :)

# Basic normalization với các tùy chọn
normalized = normalizer.basic_normalizer(
    text,
    case_folding=True,
    mode='lower',
    remove_emoji=False,
    split_emoji=True
)
print(normalized)
# Output: ['hôm', 'nay', 'tôi', 'rất', 'vui', '😊', 'và', 'hạnh', 'phúc', '🎉', '!']

# Loại bỏ emoji
normalized_no_emoji = normalizer.basic_normalizer(
    text,
    case_folding=True,
    remove_emoji=True
)
print(normalized_no_emoji)
# Output: ['hôm', 'nay', 'tôi', 'rất', 'vui', 'và', 'hạnh', 'phúc', '!']
```

### 2. 😊 EmojiHandler — Xử lý emoji

```python
from visonorm import EmojiHandler

# Khởi tạo EmojiHandler
emoji_handler = EmojiHandler()

text = "Hôm nay tôi rất vui 😊🎉😊 và hạnh phúc 🎉!"

# Detect emojis
emojis = emoji_handler.detect_emoji(text)
print(f"Detected emojis: {emojis}")
# Output: Detected emojis: ['😊🎉😊', '🎉']

# Split emoji text
split_text = emoji_handler.split_emoji_text(text)
print(f"Split emoji text: {split_text}")
# Output: Hôm nay tôi rất vui 😊 🎉 😊 và hạnh phúc 🎉 !

# Split consecutive emojis
text_consecutive = "Hôm nay tôi rất vui 😊🎉😊"
split_consecutive = emoji_handler.split_emoji_emoji(text_consecutive)
print(f"Split consecutive: {split_consecutive}")
# Output: Hôm nay tôi rất vui 😊 🎉 😊

# Remove emojis
text_no_emoji = emoji_handler.remove_emojis(text)
print(f"Text without emojis: {text_no_emoji}")
# Output: Hôm nay tôi rất vui và hạnh phúc !
```

### 3. ✏️ Lexical Normalization — Chuẩn hóa văn bản mạng xã hội

#### Sử dụng ViSoLexNormalizer

```python
from visonorm import ViSoLexNormalizer

# Khởi tạo với model mặc định (hadung1802/vit5-base-normalizer-mix100)
normalizer = ViSoLexNormalizer()

# Hoặc chỉ định model cụ thể từ HuggingFace
# normalizer = ViSoLexNormalizer(model_repo="hadung1802/visobert-normalizer-mix100")
# normalizer = ViSoLexNormalizer(model_repo="hadung1802/bartpho-normalizer-mix100")

# Chuẩn hóa câu
input_str = "sv dh gia dinh chua cho di lam :))"
normalized = normalizer.normalize_sentence(input_str)
print(f"Original: {input_str}")
print(f"Normalized: {normalized}")
# Output:
# Original: sv dh gia dinh chua cho di lam :))
# Normalized: sinh viên đại học gia đình chưa cho đi làm :))

# Chuẩn hóa và phát hiện NSW cùng lúc
nsw_spans, normalized_text = normalizer.normalize_sentence(input_str, detect_nsw=True)
print(f"Normalized: {normalized_text}")
print("Detected NSW:")
for nsw in nsw_spans:
    print(f"  - '{nsw['nsw']}' → '{nsw['prediction']}' (confidence: {nsw['confidence_score']})")
# Output:
# Normalized: sinh viên đại học gia đình chưa cho đi làm :))
# Detected NSW:
#   - 'sv' → 'sinh viên' (confidence: 1.0)
#   - 'dh' → 'đại học' (confidence: 1.0)
#   - 'dinh' → 'đình' (confidence: 1.0)
#   - 'chua' → 'chưa' (confidence: 1.0)
#   - 'di' → 'đi' (confidence: 1.0)
#   - 'lam' → 'làm' (confidence: 1.0)
```

#### Sử dụng NswDetector

```python
from visonorm import NswDetector

# Khởi tạo detector
detector = NswDetector()

# Phát hiện NSW
input_str = "sv dh gia dinh chua cho di lam"
nsw_spans = detector.detect_nsw(input_str)
for nsw in nsw_spans:
    print(f"NSW: '{nsw['nsw']}' → '{nsw['prediction']}' (confidence: {nsw['confidence_score']})")
```

#### Sử dụng hàm tiện ích

```python
from visonorm import detect_nsw, normalize_sentence

# Phát hiện NSW
nsw_spans = detect_nsw("sv dh gia dinh chua cho di lam")

# Chuẩn hóa câu
normalized = normalize_sentence("sv dh gia dinh chua cho di lam")

# Chuẩn hóa và phát hiện NSW
nsw_spans, normalized = normalize_sentence("sv dh gia dinh chua cho di lam", detect_nsw=True)
```

### 4. 📊 Resource Management — Quản lý dataset

Các dataset được lưu trữ trên **GitHub Releases** và tự động tải về khi cần.

```python
from visonorm import list_datasets, load_dataset, get_dataset_info

# Liệt kê tất cả datasets có sẵn
datasets = list_datasets()
print("Available datasets:")
for i, dataset in enumerate(datasets, 1):
    print(f"{i}. {dataset}")

# Lấy thông tin chi tiết về một dataset
info = get_dataset_info("ViLexNorm")
print(f"URL: {info['url']}")
print(f"Type: {info['type']}")

# Tải dataset (tự động cache)
df = load_dataset("ViLexNorm")
print(f"Dataset shape: {df.shape}")
print(df.head())

# Force download lại dataset
df = load_dataset("ViLexNorm", force_download=True)
```

**Các datasets có sẵn:**

- **ViLexNorm**: Vietnamese Lexical Normalization Dataset
- **ViHSD**: Vietnamese Hate Speech Detection Dataset
- **ViHOS**: Vietnamese Hate and Offensive Speech Dataset
- **UIT-VSMEC**: Vietnamese Social Media Emotion Corpus
- **ViSpamReviews**: Vietnamese Spam Review Detection Dataset
- **UIT-ViSFD**: Vietnamese Sentiment and Emotion Detection Dataset
- **UIT-ViCTSD**: Vietnamese Customer Review Sentiment Dataset
- **ViTHSD**: Vietnamese Toxic Hate Speech Detection Dataset
- **BKEE**: Vietnamese Emotion Recognition Dataset
- **UIT-ViQuAD**: Vietnamese Question Answering Dataset

### 5. 🧠 Task Models — Mô hình xử lý tác vụ

Tất cả các mô hình task được lưu trữ trên **HuggingFace Hub** tại [https://huggingface.co/visolex](https://huggingface.co/visolex).

#### SpamReviewDetection — Phát hiện spam

```python
from visonorm import SpamReviewDetection

# Xem danh sách các model có sẵn
models = SpamReviewDetection.list_models()
print("Available models:", SpamReviewDetection.list_model_names())

# Khởi tạo với model phobert-v1 (binary classification)
spam_detector = SpamReviewDetection("phobert-v1")

# Hoặc sử dụng các model khác
# spam_detector = SpamReviewDetection("phobert-v1-multiclass")  # Multiclass model

# Phát hiện spam
text = "Sản phẩm rất tốt, chất lượng cao!"
result = spam_detector.predict(text)
print(f"Text: {text}")
print(f"Result: {result}")
# Output: Result: Non-spam
```

#### HateSpeechDetection — Phát hiện hate speech

```python
from visonorm import HateSpeechDetection

# Xem danh sách các model có sẵn
print("Available models:", HateSpeechDetection.list_model_names())

# Khởi tạo detector
hate_detector = HateSpeechDetection("phobert-v1")
# Hoặc: HateSpeechDetection("phobert-v2"), HateSpeechDetection("visobert"), etc.

# Phát hiện hate speech
text = "Văn bản cần kiểm tra hate speech"
result = hate_detector.predict(text)
print(f"Result: {result}")
# Output: Result: CLEAN
```

#### HateSpeechSpanDetection — Phát hiện span của hate speech

```python
from visonorm import HateSpeechSpanDetection

# Xem danh sách các model có sẵn
print("Available models:", HateSpeechSpanDetection.list_model_names())

# Khởi tạo detector
hate_span_detector = HateSpeechSpanDetection("phobert-v1")
# Hoặc: HateSpeechSpanDetection("vihate-t5"), HateSpeechSpanDetection("visobert"), etc.

# Phát hiện span
text = "Nói cái lồn gì mà khó nghe"
result = hate_span_detector.predict(text)
print(f"Result: {result}")
# Output: {'tokens': [...], 'text': '...'}
```

#### EmotionRecognition — Nhận diện cảm xúc

```python
from visonorm import EmotionRecognition

# Xem danh sách các model có sẵn
print("Available models:", EmotionRecognition.list_model_names())

# Khởi tạo detector
emotion_detector = EmotionRecognition("phobert-v2")
# Hoặc: EmotionRecognition("phobert-v1"), EmotionRecognition("visobert"), etc.

# Nhận diện cảm xúc
text = "Tôi rất vui mừng và hạnh phúc!"
emotion = emotion_detector.predict(text)
print(f"Emotion: {emotion}")
# Output: Emotion: Enjoyment
```

#### AspectSentimentAnalysis — Phân tích sentiment theo aspect

```python
from visonorm import AspectSentimentAnalysis

# Xem danh sách các domain có sẵn
print("Available domains:", AspectSentimentAnalysis.list_domains())

# Xem danh sách các model cho một domain cụ thể
print("Models for smartphone:", AspectSentimentAnalysis.list_model_names("smartphone"))
print("Models for restaurant:", AspectSentimentAnalysis.list_model_names("restaurant"))
print("Models for hotel:", AspectSentimentAnalysis.list_model_names("hotel"))

# Khởi tạo với domain smartphone và model phobert
absa = AspectSentimentAnalysis("smartphone", "phobert")
# Hoặc sử dụng các model khác: "phobert-v2", "bartpho", "vit5", "visobert", etc.

# Hoặc các domain khác
# absa = AspectSentimentAnalysis("restaurant", "phobert-v1")
# absa = AspectSentimentAnalysis("hotel", "phobert-v1")

# Phân tích sentiment
text = "Điện thoại có camera rất tốt nhưng pin nhanh hết"
aspects = absa.predict(text, threshold=0.25)
print(f"Aspects: {aspects}")
# Output: [('BATTERY', 'neutral'), ('FEATURES', 'neutral'), ('PERFORMANCE', 'positive'), ...]
```

### 6. 🎯 Advanced Usage — Sử dụng nâng cao

#### Kết hợp nhiều chức năng

```python
from visonorm import BasicNormalizer, EmojiHandler, ViSoLexNormalizer

def process_text_advanced(text):
    """Xử lý văn bản với nhiều bước"""
    print(f"Original text: {text}")
    
    # Bước 1: Xử lý emoji
    emoji_handler = EmojiHandler()
    emojis = emoji_handler.detect_emoji(text)
    print(f"Detected emojis: {emojis}")
    
    # Bước 2: Chuẩn hóa cơ bản
    normalizer = BasicNormalizer()
    normalized = normalizer.basic_normalizer(text, case_folding=True)
    print(f"Basic normalized: {normalized}")
    
    # Bước 3: Chuẩn hóa lexical với deep learning
    lex_normalizer = ViSoLexNormalizer()
    final_normalized = lex_normalizer.normalize_sentence(text)
    print(f"Lexical normalized: {final_normalized}")
    
    return {
        'original': text,
        'emojis': emojis,
        'basic_normalized': normalized,
        'lexical_normalized': final_normalized
    }

# Test
result = process_text_advanced("Hôm nay tôi rất😊 VUI 😊😊 và HẠNH PHÚC!")
```

---

## 🌐 Resources

### HuggingFace Hub

Tất cả các mô hình và resources được publish trên HuggingFace Hub:

- **Organization**: [https://huggingface.co/visolex](https://huggingface.co/visolex)
- **Models**: Xem danh sách đầy đủ tại [https://huggingface.co/visolex](https://huggingface.co/visolex)

**Các mô hình normalization có sẵn:**

- `visolex/visobert-normalizer-mix100` (mặc định)


### GitHub Releases

Các datasets được lưu trữ dưới dạng GitHub Releases và tự động tải về khi sử dụng:

- **Repository**: [https://github.com/AnhHoang0529/visonorm](https://github.com/AnhHoang0529/visonorm)
- **Releases**: [https://github.com/AnhHoang0529/visonorm/releases](https://github.com/AnhHoang0529/visonorm/releases)

---

## 📖 API Reference

### Core Components

#### BasicNormalizer

```python
normalizer = BasicNormalizer()

# Methods
normalizer.case_folding(text, mode='lower')  # 'lower', 'upper', 'capitalize'
normalizer.tone_normalization(text)
normalizer.remove_redundant_dots(text)
normalizer.remove_emojis(text)
normalizer.basic_normalizer(
    text,
    case_folding=True,
    mode='lower',
    tone_normalization=True,
    remove_emoji=False,
    split_emoji=True
)
```

#### EmojiHandler

```python
emoji_handler = EmojiHandler()

# Methods
emoji_handler.detect_emoji(text)
emoji_handler.split_emoji_text(text)
emoji_handler.split_emoji_emoji(text)
emoji_handler.remove_emojis(text)
```

#### ViSoLexNormalizer

```python
normalizer = ViSoLexNormalizer(model_repo=None, device='cpu')

# Methods
normalizer.normalize_sentence(input_str, detect_nsw=False)
```

#### NswDetector

```python
detector = NswDetector(model_repo=None, device='cpu')

# Methods
detector.detect_nsw(input_str)
detector.concatenate_nsw_spans(nsw_spans)
```

---

## 🔬 Examples

Xem file [test_toolkit.ipynb](test_toolkit.ipynb) để có các ví dụ chi tiết và đầy đủ hơn.

---

## 📝 Citation

Nếu bạn sử dụng ViSoNorm trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@misc{visonorm2024,
  title={ViSoNorm: Vietnamese Social Media Lexical Normalization Toolkit},
  author={Ha Dung Nguyen},
  year={2024},
  url={https://github.com/AnhHoang0529/visonorm},
  note={Available at https://huggingface.co/visolex}
}
```

---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Anh Thi-Hoang Nguyen** - *Maintainer* - [anhnth@uit.edu.vn](mailto:anhnth@uit.edu.vn)
- **Ha Dung Nguyen** - *Maintainer* - [dungngh@uit.edu.vn](mailto:dungngh@uit.edu.vn)

---

## 🙏 Acknowledgments

- HuggingFace for hosting models and providing the transformers library
- The Vietnamese NLP community for datasets and feedback

---

## 📞 Contact & Support

- **GitHub Issues**: [https://github.com/AnhHoang0529/visonorm/issues](https://github.com/AnhHoang0529/visonorm/issues)
- **Email**: anhnth@uit.edu.vn
- **HuggingFace**: [https://huggingface.co/visolex](https://huggingface.co/visolex)

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/AnhHoang0529/visonorm](https://github.com/AnhHoang0529/visonorm)
- **PyPI Package**: [https://pypi.org/project/visonorm/](https://pypi.org/project/visonorm/)
- **HuggingFace Hub**: [https://huggingface.co/visolex](https://huggingface.co/visolex)
- **Documentation**: [https://github.com/AnhHoang0529/visonorm](https://github.com/AnhHoang0529/visonorm)
