import flask
import re
import string
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import torch

app = flask.Flask(__name__)

def remove_abb(data):
    data = re.sub(r"\bko\b", "không", data)
    data = re.sub(r"\bmk\b", "mình", data)
    data = re.sub(r"\be\b", "em", data)
    data = re.sub(r"\biu\b", "yêu", data)
    data = re.sub(r"\bb\b", "bạn", data)
    data = re.sub(r"\bad\b", "admin", data)
    data = re.sub(r"\bng\b", "người", data)
    data = re.sub(r"\bhh\b", "học hộ", data)
    data = re.sub(r"\bcnxh\b", "chủ nghĩa xã hội", data)
    data = re.sub(r"\bbhyt\b", "bảo hiểm y tế", data)
    data = re.sub(r"\bbt\b", "biết", data)
    data = re.sub(r"\bib\b", "inbox", data)
    data = re.sub(r"\bin4\b", "info", data)
    data = re.sub(r"\bktx\b", "kí túc xá", data)
    data = re.sub(r"\bny\b", "người yêu", data)
    data = re.sub(r"\bvc\b", "việc", data)
    data = re.sub(r"\btrg\b", "trường", data)
    data = re.sub(r"\bfb\b", "facebook", data)
    data = re.sub(r"\bmn\b", "mọi người", data)
    data = re.sub(r"\bib\b", "inbox", data)
    data = re.sub(r"\bqldt\b", "quản lý đào tạo", data)
    data = re.sub(r"\bk\b", "không", data)
    data = re.sub(r"\bđc\b", "được", data)
    data = re.sub(r"\bak\b", "ạ", data)
    data = re.sub(r"\bm\b", "mình", data)
    data = re.sub(r"\banqp\b", "an ninh quốc phòng", data)
    data = re.sub(r"\bgdtc\b", "giáo dục thể chất", data)
    data = re.sub(r"\bbb\b", "bạn bè", data)
    data = re.sub(r"\ba.c\b", "anh chị", data)
    data = re.sub(r"\bac\b", "anh chị", data)
    data = re.sub(r"\bng\b", "người", data)
    data = re.sub(r"\bntnt\b", "như thế nào", data)
    data = re.sub(r"\br\b", "rồi", data)
    data = re.sub(r"\bhok\b", "học", data)
    data = re.sub(r"\btp\b", "thành phần", data)
    data = re.sub(r"\bh\b", "giờ", data)
    data = re.sub(r"\bace\b", "anh chị em", data)
    data = re.sub(r"\bbâyh\b", "bây giờ", data)
    data = re.sub(r"\bm.n\b", "mọi người", data)
    data = re.sub(r"\ba/c\b", "anh chị", data)
    data = re.sub(r"\btl\b", "trả lời", data)
    data = re.sub(r"\bbh\b", "bây giờ", data)
    data = re.sub(r"\ba\b", "anh", data)
    data = re.sub(r"\ba/c/e\b", "anh chị em", data)
    data = re.sub(r"\bxl\b", "xin lỗi", data)
    data = re.sub(r"\bô\b", "ông", data)
    data = re.sub(r"\bgd\b", "gia đình", data)
    data = re.sub(r"\bđki\b", "đăng ký", data)
    data = re.sub(r"\bcmt\b", "bình luận", data)
    data = re.sub(r"\bnt\b", "nhắn tin", data)
    data = re.sub(r"\bhk\b", "học kỳ", data)
    data = re.sub(r"\btnao\b", "thế nào", data)
    data = re.sub(r"\bt.p\b", "thành phần", data)
    data = re.sub(r"\bdstt\b", "đại số tuyến tính", data)
    data = re.sub(r"\bvly\b", "vật lý", data)
    data = re.sub(r"\bbtl\b", "bài tập lớn", data)
    data = re.sub(r"\bhnay\b", "hôm nay", data)
    data = re.sub(r"\bsv\b", "sinh viên", data)
    data = re.sub(r"\bsp\b", "hỗ trợ", data)
    data = re.sub(r"\bdkk\b", "được", data)
    data = re.sub(r"\brr\b", "rời rạc", data)
    data = re.sub(r"\biem\b", "em", data)
    data = re.sub(r"\bclb\b", "câu lạc bộ", data)
    data = re.sub(r"\bmik\b", "mình", data)
    data = re.sub(r"\bcfs\b", "confession", data)
    data = re.sub(r"\bwen\b", "quen", data)
    data = re.sub(r"\bcc\b", "chứng chỉ", data)
    return data

def remove_emoji(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def preprocessData(x):
    x = x.lower() if isinstance(x, str) else x
    if isinstance(x, str):
        for i in string.punctuation:
            if i in x:
                x = x.replace(i,'')
    x = remove_emoji(x)
    x = remove_abb(x)
    return x

check_point = torch.load('D:\\CPP\AI_ML\\NCKH\Project\\tokenize\\ten_file.pth')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

model = AutoModelForSequenceClassification.from_pretrained('vinai/phobert-base')
model.config.max_position_embeddings = 256
optimizer = AdamW(model.parameters(), lr=2e-5)


model.load_state_dict(check_point)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


@app.post('/predict')
def predict():
    text = flask.request.json.get("text")

    text = preprocessData(text)
    text = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=256, return_tensors='pt')

    input_ids = text['input_ids']
    attention_mask = text['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()

    # Trả về kết quả dự đoán dưới dạng JSON
    return flask.jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)