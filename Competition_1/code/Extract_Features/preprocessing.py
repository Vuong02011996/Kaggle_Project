import re
import spacy
import polars as pl
import string


"""
spacy.load("en_core_web_sm") tải mô hình ngôn ngữ tiếng Anh nhỏ của SpaCy (small English model).
"""
nlp = spacy.load("en_core_web_sm")
with open('/home/oryza/Desktop/KK/Competition_1/data/1_learning-agency-lab-automated-essay-scoring-2/words.txt', 'r') as file:
    """
    Đọc từng từ trong tệp, loại bỏ khoảng trắng ở đầu và cuối, và chuyển tất cả từ thành chữ thường.
    Lưu tất cả các từ vào một tập hợp (set) có tên english_vocab.
    """
    english_vocab = set(word.strip().lower() for word in file)


def count_spelling_errors(text):
    """

    :param text:
    :return: Trả về số lượng lỗi chính tả.
    """

    """
    doc = nlp(text): Sử dụng đối tượng nlp để phân tích đoạn văn bản, tạo ra một đối tượng doc chứa các token (từ, dấu câu, v.v.).
    """
    doc = nlp(text)

    """
    lemmatized_tokens = [token.lemma_.lower() for token in doc]: Lấy các lemma (dạng gốc) của các token, 
    chuyển thành chữ thường và lưu vào danh sách lemmatized_tokens.
     Lemma là dạng cơ bản của một từ, ví dụ "running" có lemma là "run".
    """
    lemmatized_tokens = [token.lemma_.lower() for token in doc]

    """
    spelling_errors = sum(1 for token in lemmatized_tokens if token not in english_vocab): 
    Đếm số lượng token trong lemmatized_tokens không có trong english_vocab, tức là những từ này bị coi là lỗi chính tả.


    Biểu thức: 1 for token in lemmatized_tokens if token not in english_vocab tạo ra một chuỗi các số 1 
    cho mỗi token không có trong english_vocab.

    Đây là một biểu thức generator (generator expression) tạo ra một iterable theo cú pháp:
    (expression for item in iterable if condition)
     sum(iterable)
    """
    spelling_errors = sum(1 for token in lemmatized_tokens if token not in english_vocab)
    return spelling_errors


def remove_punctuation(text):
    """
    Remove all punctuation(cham cau) from the input text.
    Loại bỏ tất cả các dấu câu khỏi một chuỗi văn bản đầu vào

    Args:
    - text (str): The input text.

    Returns:
    - str: The text with punctuation removed.
    """

    """
    string.punctuation là một chuỗi chứa tất cả các ký tự dấu câu trong Python (ví dụ: !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`).
    str.maketrans('', '', string.punctuation) tạo ra một bảng dịch 
    mà trong đó tất cả các ký tự trong string.punctuation sẽ được ánh xạ đến None, nghĩa là sẽ bị loại bỏ
    """
    translator = str.maketrans('', '', string.punctuation)

    """
    text.translate(translator) sử dụng bảng dịch translator để loại bỏ tất cả các ký tự dấu câu trong chuỗi văn bản text
    """
    return text.translate(translator)


"""
cList: Một từ điển (dictionary) chứa các từ viết tắt làm khóa (keys) và các từ mở rộng tương ứng làm giá trị (values).
"""
cList = {
  "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
  "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is",
  "how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have",
  "isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not",
  "might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
  "shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is",
  "should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we had",
  "we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
  "weren't": "were not","what'll": "what will","what'll've": "what will have",
  "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
  "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will","who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is",
  "why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
  "wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would",
  "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you had","you'd've": "you would have","you'll": "you you will","you'll've": "you you will have","you're": "you are",  "you've": "you have"
   }

"""
re.compile là một hàm trong module re của Python, được sử dụng để biên dịch một chuỗi biểu thức chính quy thành một đối tượng pattern. 
Đối tượng này có thể được sử dụng để tìm kiếm các mẫu khớp trong văn bản một cách hiệu quả.

cList.keys() trả về một đối tượng view chứa tất cả các khóa (từ viết tắt) trong từ điển cList

join là một phương thức của chuỗi, nó nối các phần tử trong iterable (ở đây là danh sách các từ viết tắt) 
thành một chuỗi duy nhất, sử dụng ký tự '|' làm dấu phân cách.
'|'.join(cList.keys()) tạo ra một chuỗi mới, trong đó tất cả các từ viết tắt được nối lại với nhau, ngăn cách bởi dấu gạch đứng |.
 Dấu gạch đứng | trong biểu thức chính quy có nghĩa là "hoặc" (OR), vì vậy chuỗi này sẽ khớp với bất kỳ từ viết tắt nào trong danh sách.

'%s' % '|'.join(cList.keys())
Biểu thức này sử dụng toán tử định dạng chuỗi % của Python. Cụ thể, nó chèn chuỗi được tạo bởi '|'.join(cList.keys()) vào vị trí của %s.

re.compile('(%s)' % '|'.join(cList.keys())) tạo ra một đối tượng pattern từ chuỗi biểu thức chính quy. 
Biểu thức này sẽ khớp với bất kỳ từ viết tắt nào trong cList.
"""
c_re = re.compile('(%s)' % '|'.join(cList.keys()))


def expandContractions(text, c_re=c_re):
    """
    Hàm expandContractions trong đoạn code được sử dụng để mở rộng các từ viết tắt trong văn bản thành các cụm từ đầy đủ tương ứng
    :param text:  Chuỗi văn bản đầu vào mà bạn muốn mở rộng các từ viết tắt.
    :param c_re: Biểu thức chính quy (regular expression) đã được biên dịch để khớp với các từ viết tắt trong cList
    :return:
    """
    def replace(match):
        """
        Hàm replace nhận một đối tượng match làm đầu vào. Đối tượng match này là kết quả của việc tìm kiếm biểu thức chính quy trong văn bản.
        text = "I'm sure I can't do this and he won't help."
        expanded_text = expandContractions(text)
        expanded_text = "I am sure I cannot do this and he will not help."
        :param match:
        :return:
        """
        """
        cList[match.group(0)]: Sử dụng từ viết tắt để tra cứu trong từ điển cList và trả về cụm từ đầy đủ tương ứng.
        """
        return cList[match.group(0)]
    """
    c_re.sub(replace, text): Sử dụng phương thức sub của đối tượng biểu thức chính quy c_re để thay thế tất cả các từ 
    viết tắt trong text bằng cụm từ đầy đủ tương ứng.
    """
    return c_re.sub(replace, text)


def removeHTML(x):
    """
    Hàm này nhận đầu vào là một chuỗi văn bản x và loại bỏ tất cả các thẻ HTML bằng cách sử dụng biểu thức chính quy html
    :param x:
    :return:
    """
    """
    html=re.compile(r'<.*?>'): Biểu thức chính quy khớp với tất cả các thẻ HTML.
    """
    html = re.compile(r'<.*?>')

    """
    html.sub(r'',x): Thay thế tất cả các thẻ HTML trong văn bản x bằng chuỗi rỗng, loại bỏ chúng khỏi văn bản.
    """
    return html.sub(r'',x)


def dataPreprocessing(x):
    """
    dataPreprocessing là hàm chính để thực hiện toàn bộ quá trình làm sạch văn bản theo các bước đã mô tả
    :param x:
    :return:
    """
    x = x.lower()
    x = removeHTML(x)
    #  Loại bỏ tất cả các đề cập tới người dùng (username mentions) bắt đầu bằng ký tự @.
    x = re.sub("@\w+", '',x)
    # Loại bỏ tất cả các từ viết tắt chứa số (vd: '90s).
    x = re.sub("'\d+", '',x)
    #  Loại bỏ tất cả các số.
    x = re.sub("\d+", '',x)
    #  Loại bỏ tất cả các URL bắt đầu bằng http.
    x = re.sub("http\w+", '',x)
    # Thay thế các khoảng trắng liên tiếp bằng một khoảng trắng duy nhất.
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    # Thay thế các dấu chấm liên tiếp bằng một dấu chấm duy nhất.
    x = re.sub(r"\.+", ".", x)
    # Thay thế các dấu phẩy liên tiếp bằng một dấu phẩy duy nhất.
    x = re.sub(r"\,+", ",", x)
    # Loại bỏ khoảng trắng ở đầu và cuối chuỗi văn bản.
    x = x.strip()
    return x