# 💡 MarketMind

**MarketMind** is a secure Django web application that leverages **Artificial Intelligence** to predict the **marketing performance** of campaigns — especially the **Return on Investment (ROI)** — in real time.  
It provides a personalized dashboard where users can input campaign data, analyze results, and make data-driven business decisions.

---

## 🚀 Features

- 🔐 User authentication and role-based access control  
- 📊 AI model (**Random Forest**) for ROI prediction  
- ⚙️ Label Encoding for preprocessing  
- 🧠 Dynamic dashboard for real-time results  
- 💾 MySQL database integration  
- 🌐 Modern and intuitive interface  

---

## ⚙️ Installation Guide

```bash
# 1. Clone the repository
git clone https://github.com/InsafMaaloui/MarketMind.git
cd MarketMind
```bash
# 2. Create and activate a virtual environment
python -m venv venv
```bash
# On macOS/Linux
source venv/bin/activate
```bash
# On Windows
venv\Scripts\activate
```bash
# 3. Install dependencies
pip install -r requirements.txt
```bash
# 4. Configure the database (MySQL)
# Update your settings.py with your MySQL credentials:
# (Example)
# DATABASES = {
#     'default': {
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': 'marketmind_db',
#         'USER': 'root',
#         'PASSWORD': '',
#         'HOST': 'localhost',
#         'PORT': '3306',
#     }
# }

# 5. Run migrations
python manage.py makemigrations
python manage.py migrate

# 6. Launch the server
python manage.py runserver

# Then open 👉 http://127.0.0.1:8000/ in your browser.

# 📊 Example Use Case
# A marketing manager logs in to MarketMind.
# They enter campaign parameters: budget, duration, target audience, and platform.
# The AI model processes the input and returns a predicted ROI.
# The result is displayed on a dynamic dashboard, allowing data-driven decision-making.

# 📈 Future Enhancements
# 🔮 Add support for deep learning models (e.g., XGBoost, LSTM)
# 📅 Integrate time-based ROI forecasting
# 📤 Export results in PDF/Excel format
# 🌍 Deploy the app on AWS or Render

# 👩‍💻 Author
# Insaf Maaloui
# 🎓 Data Science & AI Student — TEK-UP University

