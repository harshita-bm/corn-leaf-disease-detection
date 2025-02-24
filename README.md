
## 🌽 Corn Leaf Disease Detection  

### 📌 **Overview**  
Corn crops are highly susceptible to various diseases, which can significantly impact yield and food security. Traditional disease detection methods rely on manual inspection, which is **time-consuming, expensive, and prone to human error**.  

This project implements an **AI-powered deep learning model (ResNet50V2-based CNN)** for the **automated detection of common corn leaf diseases**. The model is trained to classify three major diseases:  
- 🌿 **Northern Corn Leaf Blight (NCLB)**  
- 🍂 **Grey Leaf Spot (GLS)**  
- 🍁 **Common Rust (CR)**  
as well as **healthy** leaves.  

With this system, farmers and agricultural professionals can quickly analyze images of corn leaves and receive disease classification and treatment recommendations. 🚀  

---

### 🏆 **Key Features**  
✅ **Deep Learning-Based Detection** – Utilizes **ResNet50V2** for high-accuracy disease classification.  
✅ **Automated Image Processing** – Preprocessing techniques like **data augmentation, resizing, and normalization** improve model performance.  
✅ **Flask Web Application** – User-friendly interface for **uploading images and receiving real-time disease diagnosis**.  
✅ **Actionable Recommendations** – Provides **treatment options (chemical & organic)** based on the detected disease.  
✅ **Scalability & Sustainability** – Designed for **farmers, researchers, and agronomists**, promoting **sustainable agricultural practices**.  

---

### 🔬 **Technology Stack**  
- **🖥 Machine Learning Frameworks** – TensorFlow, Keras  
- **📸 Deep Learning Model** – ResNet50V2 CNN  
- **🛠 Backend** – Flask  
- **🌐 Deployment** – Web-based interface for image uploads  
- **📊 Data Processing** – Image augmentation, feature extraction, classification  

---

### 📂 **Dataset**  
The model is trained on a dataset of **4,188 images**, including:  
- **Northern Corn Leaf Blight**: 1,146 images  
- **Grey Leaf Spot**: 574 images  
- **Common Rust**: 1,306 images  
- **Healthy Leaves**: 1,162 images  

Data was collected from **agricultural research institutions, online repositories, and field images**.  

🔗 **Download Dataset & Model Files:** [Google Drive Link](https://drive.google.com/drive/folders/1n9wZ2lXghE9D-OljI4zc81v8Fal5wecr)  

---

### 🚀 **How It Works**  
1️⃣ **Upload** a clear image of a corn leaf on the Flask web app.  
2️⃣ The **ResNet50V2** model analyzes the image and classifies it into **one of the four categories**.  
3️⃣ The system displays the **predicted disease name and confidence score**.  
4️⃣ It suggests **organic & chemical treatments** for effective disease management.  

---

### 🔧 **Installation & Setup**  
#### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/harshita-bm/corn-leaf-disease-detection.git
cd corn-leaf-disease-detection
```
#### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```
#### **3️⃣ Run the Flask App**  
```bash
python app.py
```
Then open **http://127.0.0.1:5000/** in your browser to use the app.  

---

### 🛠 **Future Enhancements**  
✨ **Mobile App Integration** – Deploy as a **mobile-friendly app** for farmers to use on the go.  
📡 **IoT & Satellite Data** – Use **drone or IoT sensors** for real-time field monitoring.  
🌍 **Multilingual Support** – Make the app accessible to farmers globally.  

---

### **Contributions**  
Contributions are welcome! Feel free to fork the repository, create a branch, and submit a PR.  

