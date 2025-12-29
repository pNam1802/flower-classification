import os
import json
import time
import torch
import torch.nn as nn
from PIL import Image
# Thêm jsonify để tối ưu cho các API endpoint sau này (nếu cần)
from flask import Flask, request, render_template, url_for, abort, jsonify
from torchvision import models, transforms
from werkzeug.utils import secure_filename
import requests

# --- Cấu hình Flask và Thư mục Upload ---
app = Flask(__name__)


UNSPLASH_ACCESS_KEY = os.environ.get('UNSPLASH_API_KEY', 'key for UNSPLASH_ACCESS_KEY')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Giới hạn kích thước file tải lên (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Khởi tạo Mô hình và Dữ liệu (KHỐI ĐÃ SỬA CUỐI CÙNG) ---
try:
    with open('flower_names.json', 'r') as f:
        original_names_map = json.load(f)

    # BƯỚC 1: Lấy tất cả các nhãn (khóa) dưới dạng CHUỖI.
    # ['1', '10', '100', '101', '102', '11', ..., '99']
    all_labels_str = list(original_names_map.keys())

    # BƯỚC 2: Sắp xếp các chuỗi này theo thứ tự TỪ ĐIỂN (Lexicographical Order).
    # Đây là cách PyTorch.ImageFolder sắp xếp tên thư mục.
    # Sẽ ra thứ tự: ['1', '10', '100', '101', '102', '11', '12', ...]
    sorted_lexicographical_labels = sorted(all_labels_str)

    # BƯỚC 3: TẠO ÁNH XẠ MỚI: Chỉ mục PyTorch (0, 1, 2, ...) tới TÊN HOA
    class_names = {}
    for i, label_str in enumerate(sorted_lexicographical_labels):
        # Chỉ mục 0 của PyTorch sẽ ánh xạ đến tên hoa của nhãn '1'
        # Chỉ mục 1 của PyTorch sẽ ánh xạ đến tên hoa của nhãn '10'
        # Chỉ mục 2 của PyTorch sẽ ánh xạ đến tên hoa của nhãn '100'
        class_names[str(i)] = original_names_map[label_str]

    # Lấy danh sách tên hoa độc nhất, sắp xếp để dùng cho Gallery
    FLOWER_NAMES_LIST = sorted(list(class_names.values()))

    print(f"Ánh xạ tên lớp đã được sửa thành công theo thứ tự Lexicographical.")

except FileNotFoundError:
    print("Error: flower_names.json not found. Classification will fail.")
    class_names = {str(i): f"Unknown Flower {i}" for i in range(102)}
    FLOWER_NAMES_LIST = [f"Unknown Flower {i}" for i in range(102)]  # Fallback
except Exception as e:
    print(f"LỖI FATAL khi tải/ánh xạ JSON: {e}")
    class_names = {str(i): f"Unknown Flower {i}" for i in range(102)}
    FLOWER_NAMES_LIST = [f"Unknown Flower {i}" for i in range(102)]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    # Tải mô hình đã lưu
    model = torch.load('flower_classifier_full.pth', map_location=device)
except Exception:
    # Nếu file pth không phải là mô hình hoàn chỉnh, thực hiện tái tạo và tải state_dict
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 102)
    )
    try:
        model.load_state_dict(torch.load('flower_classifier_full.pth', map_location=device))
    except Exception as e:
        print(f"FATAL ERROR: Could not load model state dictionary. {e}")
        
model = model.to(device)
model.eval()

# Preprocessing Transforms
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dữ liệu Ánh xạ và Fallback (Giữ nguyên) ---
flower_name_mapping = {
    "pink primrose": "Primula_vulgaris", "hard-leaved pocket orchid": "Caladenia",
    "canterbury bells": "Campanula_medium", "sweet pea": "Lathyrus_odoratus",
    "english marigold": "Calendula_officinalis", "tiger lily": "Lilium_lancifolium",
    "moon orchid": "Phalaenopsis_amabilis", "bird of paradise": "Strelitzia_reginae",
    "monkshood": "Aconitum", "globe thistle": "Echinops", "snapdragon": "Antirrhinum_majus",
    "colt's foot": "Tussilago_farfara", "king protea": "Protea_cynaroides",
    "spear thistle": "Cirsium_vulgare", "yellow iris": "Iris_pseudacorus",
    "globe-flower": "Trollius_europaeus", "purple coneflower": "Echinacea_purpurea",
    "peruvian lily": "Alstroemeria", "balloon flower": "Platycodon_grandiflorus",
    "giant white arum lily": "Zantedeschia_aethiopica", "fire lily": "Lilium_bulbiferum",
    "pincushion flower": "Scabiosa", "fritillary": "Fritillaria", "red ginger": "Alpinia_purpurata",
    "grape hyacinth": "Muscari", "corn poppy": "Papaver_rhoeas", "prince of wales feathers": "Amaranthus_hypochondriacus",
    "stemless gentian": "Gentiana_acaulis", "artichoke": "Cynara_scolymus",
    "sweet william": "Dianthus_barbarus", "carnation": "Dianthus_caryophyllus",
    "garden phlox": "Phlox_paniculata", "love in the mist": "Nigella_damascena",
    "mexican aster": "Cosmos_bipinnatus", "alpine sea holly": "Eryngium_alpinum",
    "ruby-lipped cattleya": "Cattleya", "cape flower": "Clivia_miniata",
    "great masterwort": "Astrantia_major", "siam tulip": "Curcuma_alismatifolia",
    "lenten rose": "Helleborus_orientalis", "barbeton daisy": "Gerbera_jamesonii",
    "daffodil": "Narcissus", "sword lily": "Gladiolus", "poinsettia": "Euphorbia_pulcherrima",
    "bolero deep blue": "Delphinium", "wallflower": "Erysimum", "marigold": "Tagetes",
    "buttercup": "Ranunculus", "oxeye daisy": "Leucanthemum_vulgare", "common dandelion": "Taraxacum_officinale",
    "petunia": "Petunia", "wild pansy": "Viola_tricolor", "primula": "Primula_vulgaris",
    "sunflower": "Helianthus_annuus", "pelargonium": "Pelargonium", "bishop of llandaff": "Dahlia",
    "gaura": "Oenothera", "geranium": "Pelargonium", "orange dahlia": "Dahlia",
    "pink-yellow dahlia": "Dahlia", "cautleya spicata": "Cautleya_spicata", "japanese anemone": "Anemone_hupehensis",
    "black-eyed susan": "Rudbeckia_hirta", "silverbush": "Convolvulus_cneorum", "californian poppy": "Eschscholzia_californica",
    "osteospermum": "Osteospermum", "spring crocus": "Crocus", "bearded iris": "Iris_germanica",
    "windflower": "Anemone_nemorosa", "tree poppy": "Dendromecon", "gazania": "Gazania",
    "azalea": "Rhododendron", "water lily": "Nymphaea", "rose": "Rosa", "thorn apple": "Datura_stramonium",
    "morning glory": "Ipomoea", "passion flower": "Passiflora", "lotus": "Nelumbo_nucifera",
    "toad lily": "Tricyrtis", "anthurium": "Anthurium", "frangipani": "Plumeria",
    "clematis": "Clematis", "hibiscus": "Hibiscus", "columbine": "Aquilegia", "desert-rose": "Adenium_obesum",
    "tree mallow": "Lavatera", "magnolia": "Magnolia", "cyclamen": "Cyclamen", "watercress": "Nasturtium_officinale",
    "canna lily": "Canna", "hippeastrum": "Hippeastrum", "bee balm": "Monarda", "ball moss": "Tillandsia_recurvata",
    "foxglove": "Digitalis", "bougainvillea": "Bougainvillea", "camellia": "Camellia", "mallow": "Malva",
    "mexican petunia": "Ruellia_simplex", "bromelia": "Bromelia", "blanket flower": "Gaillardia",
    "trumpet creeper": "Campsis_radicans", "blackberry lily": "Belamcanda_chinensis"
}

flower_info_fallback = {
    "siam tulip": "Curcuma alismatifolia, also known as Siam tulip or summer tulip, is a tropical plant native to Laos, northern Thailand, and Cambodia. Despite its name, it is not related to the tulip but is a close relative of turmeric.",
    "monkshood": "Aconitum, also known as monkshood or wolfsbane, is a genus of over 250 species of flowering plants. These toxic perennials grow in mountainous regions of the Northern Hemisphere.",
    "bolero deep blue": "Delphinium, commonly known as larkspur, is a genus of about 300 species of flowering plants, native to the Northern Hemisphere and high mountains of Africa. They are known for their dolphin-shaped flowers."
}

wikipedia_cache = {}
def load_cache():
    global wikipedia_cache
    if os.path.exists('wikipedia_cache.json'):
        try:
            with open('wikipedia_cache.json', 'r') as f:
                wikipedia_cache = json.load(f)
        except Exception:
            pass
def save_cache():
    try:
        with open('wikipedia_cache.json', 'w') as f:
            json.dump(wikipedia_cache, f)
    except Exception:
        pass
load_cache()

def get_wikipedia_info(flower_name: str) -> str:
    """Lấy thông tin tóm tắt về loài hoa từ Wikipedia (sử dụng cache)."""
    # (Hàm này được giữ nguyên logic gọi API, timeout và fallback)
    flower_name_lower = flower_name.lower()
    if flower_name_lower in wikipedia_cache:
        return wikipedia_cache[flower_name_lower]
    
    wiki_name = flower_name_mapping.get(flower_name_lower, flower_name_lower).replace(' ', '_')
    headers = {'User-Agent': 'FlowerClassifierApp/1.0 (nguyenphuongnam22114@gmail.com)'}
    
    for attempt in range(3):
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_name}"
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                info = response.json().get('extract', f"No information available for {flower_name_lower.title()}.")
                wikipedia_cache[flower_name_lower] = info
                save_cache()
                return info
            
            elif response.status_code in [429, 403]:
                time.sleep(2 ** attempt)
            else:
                break
        
        except Exception:
            time.sleep(1)
            
    if flower_name_lower in flower_info_fallback:
        wikipedia_cache[flower_name_lower] = flower_info_fallback[flower_name_lower]
        save_cache()
        return flower_info_fallback[flower_name_lower]
    
    error_msg = f"No information available for {flower_name_lower.title()}."
    wikipedia_cache[flower_name_lower] = error_msg
    save_cache()
    return error_msg


def predict_image(image_path: str) -> list:
    """Thực hiện dự đoán và trả về Top 3 kết quả [(Tên, Độ tin cậy)]."""
    image = Image.open(image_path).convert('RGB')
    image = test_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        top_probs, top_indices = probabilities.topk(3)

    # QUAN TRỌNG: Tra cứu chỉ mục PyTorch (idx.item()) mà không cần + 1
    top_predictions = [
        (class_names[str(idx.item())], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return top_predictions
def get_related_images(top_flower: str, num_images: int = 4) -> list:
    """Lấy hình ảnh liên quan từ Unsplash (sử dụng Access Key)."""
    # Xử lý trường hợp không có key
    if UNSPLASH_ACCESS_KEY == 'YOUR_UNSPLASH_ACCESS_KEY_HERE':
        # Trả về placeholder nếu chưa cấu hình key
        return [f"https://placehold.co/150x150/50C878/FFFFFF?text={top_flower.title()}"] * num_images
        
    top_flower = top_flower.lower()
    if top_flower in flower_name_mapping:
        top_flower = flower_name_mapping[top_flower].lower().replace('_', ' ')
        
    url = f"https://api.unsplash.com/search/photos?query={top_flower}&per_page={num_images}&client_id={UNSPLASH_ACCESS_KEY}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            results = response.json().get('results', [])
            urls = [item['urls']['small'] for item in results if 'urls' in item and 'small' in item['urls']][:num_images]
            if urls:
                return urls
    except Exception:
        pass
        
    # Fallback cuối cùng
    return [f"https://source.unsplash.com/150x150/?{top_flower}"] * num_images
def get_unsplash_images(query, count=4):
    """Lấy ảnh từ Unsplash dựa trên từ khóa."""
    if not UNSPLASH_ACCESS_KEY:
        print("Lỗi: Chưa cung cấp UNSPLASH_ACCESS_KEY. Không thể lấy ảnh Unsplash.")
        return []

    url = f"https://api.unsplash.com/search/photos"
    params = {
        "query": f"{query} flower", # Thêm "flower" để tìm kiếm chính xác hơn
        "per_page": count,
        "client_id": UNSPLASH_ACCESS_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=5) # Thêm timeout
        response.raise_for_status() # Nâng ngoại lệ cho trạng thái lỗi HTTP
        data = response.json()
        # Lấy URL của ảnh nhỏ hơn để tải nhanh hơn
        image_urls = [result['urls']['small'] for result in data['results']]
        return image_urls
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi lấy ảnh từ Unsplash cho '{query}': {e}")
        return []
    except Exception as e:
        print(f"Lỗi không xác định khi xử lý ảnh Unsplash cho '{query}': {e}")
        return []
# --- Route Chính ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Route xử lý hiển thị trang chủ và kết quả phân loại."""
    
    context = {
        'uploaded_image': None,
        'predictions': None,
        'top_flower': None,
        'related_images': None,
        'error': None,
        'flower_info': {}
    }

    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                context['error'] = "Vui lòng chọn một tệp ảnh để tải lên."
                return render_template('index.html', **context)
            
            file = request.files['file']
            if file.filename == '':
                context['error'] = "Tệp không hợp lệ. Vui lòng chọn lại."
                return render_template('index.html', **context)
            
            # Kiểm tra định dạng file cơ bản
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                context['error'] = "Định dạng tệp không được hỗ trợ (PNG/JPEG)."
                return render_template('index.html', **context)

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            context['uploaded_image'] = url_for('static', filename=f'uploads/{filename}')
            
            # Thực hiện dự đoán
            predictions = predict_image(file_path)
            context['predictions'] = predictions
            context['top_flower'] = predictions[0][0]
            
            # Lấy ảnh liên quan và thông tin Wikipedia
            context['related_images'] = get_related_images(context['top_flower'])
            
            flower_info_dynamic = {}
            for pred in predictions:
                flower_name = pred[0]
                flower_info_dynamic[flower_name.lower()] = get_wikipedia_info(flower_name)
                time.sleep(1) # Tránh rate limit của Wikipedia
            
            context['flower_info'] = flower_info_dynamic
            
        except requests.exceptions.Timeout:
            context['error'] = "Lỗi kết nối API bên ngoài (Wikipedia/Unsplash) hoặc mạng chậm. Vui lòng thử lại."
        except Exception as e:
            # Bắt các lỗi chung khác (ML model error, IO error, etc.)
            context['error'] = f"Đã xảy ra lỗi hệ thống khi xử lý ảnh. Chi tiết: {str(e)}"
            
    return render_template('index.html', **context)

# --- Routes cho Thư viện và Về chúng tôi ---

@app.route('/gallery')
@app.route('/gallery')
def gallery():
    flower_list = FLOWER_NAMES_LIST
    # Tạo danh sách các dictionary chứa tên hoa và URL ảnh
    flowers_with_images = []
    for flower_name in flower_list:
        # Lấy 1 ảnh từ Unsplash cho mỗi loài hoa
        image_urls = get_unsplash_images(flower_name, count=1)
        flowers_with_images.append({
            'name': flower_name,
            'image': image_urls[0] if image_urls else None # Lấy ảnh đầu tiên hoặc None
        })
    return render_template('gallery.html', flowers=flowers_with_images) # Truyền list of dicts

@app.route('/about')
def about():
    """Route hiển thị thông tin về dự án, công nghệ và đội ngũ."""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
