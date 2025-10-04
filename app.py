from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'supersecretkey'  # Change this in production

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load dataset
df = pd.read_csv("Cleaned_Recipe_Dataset.csv")

# Load pre-trained Word2Vec model
df['Clean_Ingredients'] = df['Clean_Ingredients'].fillna('').astype(str)
tokenized_ingredients = [recipe.split() for recipe in df['Clean_Ingredients']]
word2vec_model = Word2Vec(sentences=tokenized_ingredients, vector_size=100, window=5, min_count=1, workers=4)

# Function to get vector representation
def get_vector(ingredients, model):
    vectors = [model.wv[word] for word in ingredients if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Compute recipe vectors
df['Vector'] = df['Clean_Ingredients'].apply(lambda x: get_vector(x.split(), word2vec_model))

def recommend_recipes(user_ingredients, df, model, top_n=5):
    user_vector = get_vector(user_ingredients, model)
    recipe_vectors = np.vstack(df['Vector'])
    
    # Compute Cosine Similarity
    similarities = cosine_similarity([user_vector], recipe_vectors)[0]
    
    # Rank recipes based on similarity scores
    df['Similarity'] = similarities
    top_recipes = df.sort_values(by="Similarity", ascending=False).head(top_n)
    return top_recipes[['Title', 'Ingredients', 'Instructions', 'Similarity']]

# Load user session
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home Route
@app.route('/')
def home():
    return render_template("index.html")

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("recommend"))
        else:
            flash("Invalid credentials, please try again.", "danger")

    return render_template("login.html")

# Register Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already taken!", "warning")
            return redirect(url_for("register"))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# Recommendation Route (Only for logged-in users)
@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    if request.method == 'POST':
        user_input = request.form['ingredients']
        num_recipes = request.form['num_recipes']

        user_ingredients = user_input.lower().split(",")  # Convert input to list

        try:
            num_recipes = int(num_recipes)
            if num_recipes <= 0:
                num_recipes = 5
        except ValueError:
            num_recipes = 5

        recommendations = recommend_recipes(user_ingredients, df, word2vec_model, num_recipes)
        return render_template("recommend.html", recipes=recommendations.to_dict(orient="records"))

    return render_template("recommend.html")

# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database
    app.run(debug=True)
