from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, MultipleFileField
from wtforms import StringField, PasswordField, TextAreaField, SelectField, SelectMultipleField, DateField, FloatField, SubmitField, HiddenField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.utils
import json
import os
import secrets
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("Warning: python-dotenv package not installed. Environment variables will not be loaded from .env file")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Email functionality setup
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
    print("Email functionality enabled")
except ImportError:
    EMAIL_AVAILABLE = False
    print("Email functionality disabled due to import issues")

# Other imports
import csv
import io
import threading
from geopy.distance import geodesic
import folium

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
# MySQL Database Configuration
# Update these values according to your XAMPP MySQL setup
MYSQL_USERNAME = 'root'  # Default XAMPP username
MYSQL_PASSWORD = ''      # Default XAMPP password (empty)
MYSQL_HOST = 'localhost' # XAMPP MySQL host
MYSQL_PORT = '3306'      # XAMPP MySQL port
MYSQL_DATABASE = 'ecotrack'  # Your database name

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)
# Security-related cookie settings
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
if os.environ.get('FLASK_ENV') == 'production':
    app.config['SESSION_COOKIE_SECURE'] = True

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Jinja filters
@app.template_filter('from_json')
def from_json_filter(value):
    """Parse a JSON string into a Python object for use in templates.

    Returns an empty list on falsy/invalid input to simplify template logic.
    """
    if not value:
        return []
    # If it's already a list (e.g., deserialized elsewhere), return as-is
    if isinstance(value, list):
        return value
    # If it's a dict representing a single image, wrap in a list
    if isinstance(value, dict):
        return [value]
    # If it's a string, try JSON first
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            # Fallback: treat as a single base64 string
            return [{'data': value}]
    return []

@app.template_filter('guess_mime')
def guess_mime_filter(filename):
    """Guess MIME type from filename; default to image/jpeg."""
    try:
        name = (filename or '').lower()
        if name.endswith('.png'):
            return 'image/png'
        if name.endswith('.gif'):
            return 'image/gif'
        if name.endswith('.webp'):
            return 'image/webp'
        # jpeg/jpg and default
        return 'image/jpeg'
    except Exception:
        return 'image/jpeg'

@app.template_filter('format_dt')
def format_dt(value, fmt='%Y-%m-%d %H:%M'):
    """Format a datetime in Asia/Manila (UTC+8) for consistent local display."""
    if not value:
        return ''
    try:
        if isinstance(value, str):
            return value
        # Treat naive timestamps as UTC and convert to UTC+8
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        dt_ph = dt.astimezone(timezone(timedelta(hours=8)))
        return dt_ph.strftime(fmt)
    except Exception:
        try:
            return value.strftime(fmt)
        except Exception:
            return ''

@app.context_processor
def inject_datetime():
    # Expose Python datetime to Jinja templates for footer year, etc.
    return {'datetime': datetime}

# Database Models (no changes needed to models)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # resident, brgy_official, admin
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    barangay = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    profile_image = db.Column(db.Text, nullable=True)  # Base64 encoded image
    email_alerts = db.Column(db.Boolean, default=True)
    sms_alerts = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    reports = db.relationship('FloodReport', backref='reporter', lazy=True)

class FloodReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    location = db.Column(db.String(200), nullable=False)
    barangay = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    flood_level = db.Column(db.String(20), nullable=False)  # low, medium, high, critical
    description = db.Column(db.Text, nullable=False)
    weather_condition = db.Column(db.String(50), nullable=True)
    flood_images = db.Column(db.Text, nullable=True)  # JSON array of base64 images
    status = db.Column(db.String(20), default='pending')  # pending, verified, resolved
    risk_score = db.Column(db.Float, nullable=True)  # AI-generated risk score
    evacuation_needed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EmergencyAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50), nullable=False)  # flood_warning, evacuation_order, weather_alert
    severity = db.Column(db.String(20), nullable=False)  # low, medium, high, critical
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    affected_barangays = db.Column(db.Text, nullable=True)  # JSON array
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    radius = db.Column(db.Float, nullable=True)  # in kilometers
    is_active = db.Column(db.Boolean, default=True)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)

class SafeZone(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(300), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    capacity = db.Column(db.Integer, nullable=False)
    barangay = db.Column(db.String(100), nullable=False)
    contact_person = db.Column(db.String(200), nullable=True)
    contact_phone = db.Column(db.String(20), nullable=True)
    facilities = db.Column(db.Text, nullable=True)  # JSON array
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PasswordResetToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(100), unique=True, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='password_reset_tokens')

class EvacuationRoute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    from_barangay = db.Column(db.String(100), nullable=False)
    to_safe_zone_id = db.Column(db.Integer, db.ForeignKey('safe_zone.id'), nullable=False)
    route_points = db.Column(db.Text, nullable=False)  # JSON array of coordinates
    distance_km = db.Column(db.Float, nullable=False)
    estimated_time = db.Column(db.Integer, nullable=False)  # in minutes
    road_conditions = db.Column(db.String(100), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    safe_zone = db.relationship('SafeZone', backref='evacuation_routes', lazy=True)

class WeatherData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    temperature = db.Column(db.Float, nullable=True)
    humidity = db.Column(db.Float, nullable=True)
    rainfall = db.Column(db.Float, nullable=True)
    pressure = db.Column(db.Float, nullable=True)
    wind_speed = db.Column(db.Float, nullable=True)
    flood_occurred = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FloodActionSuggestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    flood_level = db.Column(db.String(20), nullable=False)  # low, medium, high, critical
    risk_score_min = db.Column(db.Float, nullable=False)
    risk_score_max = db.Column(db.Float, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    actions = db.Column(db.Text, nullable=False)  # JSON array of actions
    priority = db.Column(db.String(20), nullable=False)  # low, medium, high, critical
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.before_request
def check_session_security():
    """Basic session security check"""
    if current_user.is_authenticated:
        session.permanent = True
        if 'last_activity' in session:
            last_activity = datetime.fromisoformat(session['last_activity'])
            if datetime.now() - last_activity > timedelta(hours=2):
                logout_user()
                session.clear()
                flash('Session expired. Please login again.')
                return redirect(url_for('login'))
        session['last_activity'] = datetime.now().isoformat()

    # Clean up expired password reset tokens
    PasswordResetToken.query.filter(PasswordResetToken.expires_at < datetime.utcnow()).delete()
    db.session.commit()

    # Nothing else here; cache headers applied globally below
    

# Forms - Updated with Tagudin barangays
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    role = SelectField('Role', choices=[('resident', 'Resident'), ('brgy_official', 'Barangay Official')], validators=[DataRequired()])
    barangay = SelectField('Barangay', choices=[
        ('', 'Select Barangay'),
        ('Ag-aguman', 'Ag-aguman'),
        ('Ambalayat', 'Ambalayat'),
        ('Baracbac', 'Baracbac'),
        ('Bario-an', 'Bario-an'),
        ('Baritao', 'Baritao'),
        ('Becques', 'Becques'),
        ('Bimmanga', 'Bimmanga'),
        ('Bio', 'Bio'),
        ('Bitalag', 'Bitalag'),
        ('Borono', 'Borono'),
        ('Bucao East', 'Bucao East'),
        ('Bucao West', 'Bucao West'),
        ('Cabaroan', 'Cabaroan'),
        ('Cabugbugan', 'Cabugbugan'),
        ('Cabulanglangan', 'Cabulanglangan'),
        ('Dacutan', 'Dacutan'),
        ('Dardarat', 'Dardarat'),
        ('Del Pilar', 'Del Pilar'),
        ('Farola', 'Farola'),
        ('Gabur', 'Gabur'),
        ('Garitan', 'Garitan'),
        ('Jardin', 'Jardin'),
        ('Lacong', 'Lacong'),
        ('Lantag', 'Lantag'),
        ('Las-ud', 'Las-ud'),
        ('Libtong', 'Libtong'),
        ('Lubnac', 'Lubnac'),
        ('Magsaysay', 'Magsaysay'),
        ('Malacañang', 'Malacañang'),
        ('Pacac', 'Pacac'),
        ('Pallogan', 'Pallogan'),
        ('Pudoc East', 'Pudoc East'),
        ('Pudoc West', 'Pudoc West'),
        ('Pula', 'Pula'),
        ('Quirino', 'Quirino'),
        ('Ranget', 'Ranget'),
        ('Rizal', 'Rizal'),
        ('Salvacion', 'Salvacion'),
        ('San Miguel', 'San Miguel'),
        ('Sawat', 'Sawat'),
        ('Tallaoen', 'Tallaoen'),
        ('Tampugo', 'Tampugo'),
        ('Tarangotong', 'Tarangotong')
    ])
    phone = StringField('Phone Number')
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class EmergencyAlertForm(FlaskForm):
    alert_type = SelectField('Alert Type', choices=[('flood_warning', 'Flood Warning'), ('evacuation_order', 'Evacuation Order'), ('weather_alert', 'Weather Alert')], validators=[DataRequired()])
    severity = SelectField('Severity', choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('critical', 'Critical')], validators=[DataRequired()])
    title = StringField('Alert Title', validators=[DataRequired()])
    message = TextAreaField('Alert Message', validators=[DataRequired()])
    affected_barangays = SelectMultipleField('Affected Barangays (optional)', choices=[
        ('Ag-aguman', 'Ag-aguman'),
        ('Ambalayat', 'Ambalayat'),
        ('Baracbac', 'Baracbac'),
        ('Bario-an', 'Bario-an'),
        ('Baritao', 'Baritao'),
        ('Becques', 'Becques'),
        ('Bimmanga', 'Bimmanga'),
        ('Bio', 'Bio'),
        ('Bitalag', 'Bitalag'),
        ('Borono', 'Borono'),
        ('Bucao East', 'Bucao East'),
        ('Bucao West', 'Bucao West'),
        ('Cabaroan', 'Cabaroan'),
        ('Cabugbugan', 'Cabugbugan'),
        ('Cabulanglangan', 'Cabulanglangan'),
        ('Dacutan', 'Dacutan'),
        ('Dardarat', 'Dardarat'),
        ('Del Pilar', 'Del Pilar'),
        ('Farola', 'Farola'),
        ('Gabur', 'Gabur'),
        ('Garitan', 'Garitan'),
        ('Jardin', 'Jardin'),
        ('Lacong', 'Lacong'),
        ('Lantag', 'Lantag'),
        ('Las-ud', 'Las-ud'),
        ('Libtong', 'Libtong'),
        ('Lubnac', 'Lubnac'),
        ('Magsaysay', 'Magsaysay'),
        ('Malacañang', 'Malacañang'),
        ('Pacac', 'Pacac'),
        ('Pallogan', 'Pallogan'),
        ('Pudoc East', 'Pudoc East'),
        ('Pudoc West', 'Pudoc West'),
        ('Pula', 'Pula'),
        ('Quirino', 'Quirino'),
        ('Ranget', 'Ranget'),
        ('Rizal', 'Rizal'),
        ('Salvacion', 'Salvacion'),
        ('San Miguel', 'San Miguel'),
        ('Sawat', 'Sawat'),
        ('Tallaoen', 'Tallaoen'),
        ('Tampugo', 'Tampugo'),
        ('Tarangotong', 'Tarangotong')
    ])
    latitude = HiddenField('Latitude')
    longitude = HiddenField('Longitude')
    radius = FloatField('Alert Radius (km)', default=5.0)
    expires_at = DateField('Expires On')
    submit = SubmitField('Send Alert')

class SafeZoneForm(FlaskForm):
    name = StringField('Safe Zone Name', validators=[DataRequired()])
    address = TextAreaField('Address', validators=[DataRequired()])
    latitude = HiddenField('Latitude', validators=[DataRequired()])
    longitude = HiddenField('Longitude', validators=[DataRequired()])
    capacity = StringField('Capacity', validators=[DataRequired()])
    barangay = SelectField('Barangay', choices=[
        ('Ag-aguman', 'Ag-aguman'),
        ('Ambalayat', 'Ambalayat'),
        ('Baracbac', 'Baracbac'),
        ('Bario-an', 'Bario-an'),
        ('Baritao', 'Baritao'),
        ('Becques', 'Becques'),
        ('Bimmanga', 'Bimmanga'),
        ('Bio', 'Bio'),
        ('Bitalag', 'Bitalag'),
        ('Borono', 'Borono'),
        ('Bucao East', 'Bucao East'),
        ('Bucao West', 'Bucao West'),
        ('Cabaroan', 'Cabaroan'),
        ('Cabugbugan', 'Cabugbugan'),
        ('Cabulanglangan', 'Cabulanglangan'),
        ('Dacutan', 'Dacutan'),
        ('Dardarat', 'Dardarat'),
        ('Del Pilar', 'Del Pilar'),
        ('Farola', 'Farola'),
        ('Gabur', 'Gabur'),
        ('Garitan', 'Garitan'),
        ('Jardin', 'Jardin'),
        ('Lacong', 'Lacong'),
        ('Lantag', 'Lantag'),
        ('Las-ud', 'Las-ud'),
        ('Libtong', 'Libtong'),
        ('Lubnac', 'Lubnac'),
        ('Magsaysay', 'Magsaysay'),
        ('Malacañang', 'Malacañang'),
        ('Pacac', 'Pacac'),
        ('Pallogan', 'Pallogan'),
        ('Pudoc East', 'Pudoc East'),
        ('Pudoc West', 'Pudoc West'),
        ('Pula', 'Pula'),
        ('Quirino', 'Quirino'),
        ('Ranget', 'Ranget'),
        ('Rizal', 'Rizal'),
        ('Salvacion', 'Salvacion'),
        ('San Miguel', 'San Miguel'),
        ('Sawat', 'Sawat'),
        ('Tallaoen', 'Tallaoen'),
        ('Tampugo', 'Tampugo'),
        ('Tarangotong', 'Tarangotong')
    ], validators=[DataRequired()])
    contact_person = StringField('Contact Person')
    contact_phone = StringField('Contact Phone')
    facilities = TextAreaField('Facilities (one per line)')
    submit = SubmitField('Add Safe Zone')

class FloodReportForm(FlaskForm):
    location = StringField('Location', validators=[DataRequired()])
    barangay = SelectField('Barangay', choices=[
        ('Ag-aguman', 'Ag-aguman'),
        ('Ambalayat', 'Ambalayat'),
        ('Baracbac', 'Baracbac'),
        ('Bario-an', 'Bario-an'),
        ('Baritao', 'Baritao'),
        ('Becques', 'Becques'),
        ('Bimmanga', 'Bimmanga'),
        ('Bio', 'Bio'),
        ('Bitalag', 'Bitalag'),
        ('Borono', 'Borono'),
        ('Bucao East', 'Bucao East'),
        ('Bucao West', 'Bucao West'),
        ('Cabaroan', 'Cabaroan'),
        ('Cabugbugan', 'Cabugbugan'),
        ('Cabulanglangan', 'Cabulanglangan'),
        ('Dacutan', 'Dacutan'),
        ('Dardarat', 'Dardarat'),
        ('Del Pilar', 'Del Pilar'),
        ('Farola', 'Farola'),
        ('Gabur', 'Gabur'),
        ('Garitan', 'Garitan'),
        ('Jardin', 'Jardin'),
        ('Lacong', 'Lacong'),
        ('Lantag', 'Lantag'),
        ('Las-ud', 'Las-ud'),
        ('Libtong', 'Libtong'),
        ('Lubnac', 'Lubnac'),
        ('Magsaysay', 'Magsaysay'),
        ('Malacañang', 'Malacañang'),
        ('Pacac', 'Pacac'),
        ('Pallogan', 'Pallogan'),
        ('Pudoc East', 'Pudoc East'),
        ('Pudoc West', 'Pudoc West'),
        ('Pula', 'Pula'),
        ('Quirino', 'Quirino'),
        ('Ranget', 'Ranget'),
        ('Rizal', 'Rizal'),
        ('Salvacion', 'Salvacion'),
        ('San Miguel', 'San Miguel'),
        ('Sawat', 'Sawat'),
        ('Tallaoen', 'Tallaoen'),
        ('Tampugo', 'Tampugo'),
        ('Tarangotong', 'Tarangotong')
    ], validators=[DataRequired()])
    latitude = HiddenField('Latitude')
    longitude = HiddenField('Longitude')
    flood_level = SelectField('Flood Level', choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High'), ('critical', 'Critical')], validators=[DataRequired()])
    description = TextAreaField('Description', validators=[DataRequired()])
    weather_condition = SelectField('Weather Condition', choices=[('sunny', 'Sunny'), ('cloudy', 'Cloudy'), ('rainy', 'Rainy'), ('stormy', 'Stormy')])
    flood_images = MultipleFileField('Upload Images (Max 5)', validators=[FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    submit = SubmitField('Submit Report')

class ProfileForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    barangay = SelectField('Barangay', choices=[
        ('Ag-aguman', 'Ag-aguman'),
        ('Ambalayat', 'Ambalayat'),
        ('Baracbac', 'Baracbac'),
        ('Bario-an', 'Bario-an'),
        ('Baritao', 'Baritao'),
        ('Becques', 'Becques'),
        ('Bimmanga', 'Bimmanga'),
        ('Bio', 'Bio'),
        ('Bitalag', 'Bitalag'),
        ('Borono', 'Borono'),
        ('Bucao East', 'Bucao East'),
        ('Bucao West', 'Bucao West'),
        ('Cabaroan', 'Cabaroan'),
        ('Cabugbugan', 'Cabugbugan'),
        ('Cabulanglangan', 'Cabulanglangan'),
        ('Dacutan', 'Dacutan'),
        ('Dardarat', 'Dardarat'),
        ('Del Pilar', 'Del Pilar'),
        ('Farola', 'Farola'),
        ('Gabur', 'Gabur'),
        ('Garitan', 'Garitan'),
        ('Jardin', 'Jardin'),
        ('Lacong', 'Lacong'),
        ('Lantag', 'Lantag'),
        ('Las-ud', 'Las-ud'),
        ('Libtong', 'Libtong'),
        ('Lubnac', 'Lubnac'),
        ('Magsaysay', 'Magsaysay'),
        ('Malacañang', 'Malacañang'),
        ('Pacac', 'Pacac'),
        ('Pallogan', 'Pallogan'),
        ('Pudoc East', 'Pudoc East'),
        ('Pudoc West', 'Pudoc West'),
        ('Pula', 'Pula'),
        ('Quirino', 'Quirino'),
        ('Ranget', 'Ranget'),
        ('Rizal', 'Rizal'),
        ('Salvacion', 'Salvacion'),
        ('San Miguel', 'San Miguel'),
        ('Sawat', 'Sawat'),
        ('Tallaoen', 'Tallaoen'),
        ('Tampugo', 'Tampugo'),
        ('Tarangotong', 'Tarangotong')
    ])
    phone = StringField('Phone Number')
    profile_image = FileField('Profile Picture', validators=[FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')])
    submit = SubmitField('Update Profile')

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_new_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Update Password')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Send Reset Link')

class ResetPasswordForm(FlaskForm):
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=8)])
    confirm_new_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Reset Password')

# Helper Functions
def create_admin_user():
    """Create default admin user if it doesn't exist"""
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@ecotrack.com',
            password_hash=generate_password_hash('admin123', method='pbkdf2:sha256'),
            role='admin',
            first_name='System',
            last_name='Administrator',
            barangay='',
            phone=''
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user created: username=admin, password=admin123")

def create_default_action_suggestions():
    """Create default flood action suggestions"""
    if FloodActionSuggestion.query.count() > 0:
        return  # Already exists
    
    suggestions = [
        # Low Risk (0-30)
        {
            'flood_level': 'low',
            'risk_score_min': 0,
            'risk_score_max': 30,
            'title': 'Low Risk - Monitor Situation',
            'description': 'The flood situation is currently manageable. Continue monitoring and prepare for potential changes.',
            'actions': json.dumps([
                'Stay informed through local news and weather updates',
                'Keep emergency supplies ready',
                'Monitor water levels in your area',
                'Avoid walking or driving through flooded areas',
                'Keep important documents in waterproof containers'
            ]),
            'priority': 'low'
        },
        # Medium Risk (31-60)
        {
            'flood_level': 'medium',
            'risk_score_min': 31,
            'risk_score_max': 60,
            'title': 'Medium Risk - Prepare for Evacuation',
            'description': 'The flood situation is becoming serious. Prepare for possible evacuation and take precautionary measures.',
            'actions': json.dumps([
                'Pack emergency bags with essential items',
                'Move valuable items to higher ground',
                'Turn off utilities if instructed',
                'Stay away from floodwaters',
                'Keep mobile phones charged',
                'Inform family members of your location',
                'Follow evacuation routes if advised'
            ]),
            'priority': 'medium'
        },
        # High Risk (61-80)
        {
            'flood_level': 'high',
            'risk_score_min': 61,
            'risk_score_max': 80,
            'title': 'High Risk - Evacuate Immediately',
            'description': 'The flood situation is dangerous. Evacuate to safer areas immediately and follow emergency protocols.',
            'actions': json.dumps([
                'EVACUATE IMMEDIATELY to higher ground',
                'Do not attempt to walk or drive through floodwaters',
                'Take only essential items',
                'Follow designated evacuation routes',
                'Go to nearest evacuation center',
                'Call emergency services if trapped',
                'Stay away from electrical equipment',
                'Help elderly and disabled neighbors if safe to do so'
            ]),
            'priority': 'high'
        },
        # Critical Risk (81-100)
        {
            'flood_level': 'critical',
            'risk_score_min': 81,
            'risk_score_max': 100,
            'title': 'CRITICAL RISK - Emergency Evacuation',
            'description': 'EXTREME DANGER! Immediate evacuation required. This is a life-threatening situation.',
            'actions': json.dumps([
                'EVACUATE IMMEDIATELY - DO NOT DELAY',
                'Call emergency services (911) if you cannot evacuate',
                'Move to the highest point possible',
                'Do not return to flooded areas',
                'Stay with emergency responders',
                'Use emergency shelters',
                'Follow all emergency instructions',
                'Help others only if it is safe to do so'
            ]),
            'priority': 'critical'
        }
    ]
    
    for suggestion in suggestions:
        action_suggestion = FloodActionSuggestion(**suggestion)
        db.session.add(action_suggestion)
    
    db.session.commit()
    print("Default flood action suggestions created")

def convert_image_to_base64(file):
    """Convert uploaded file to base64 string"""
    if file:
        return base64.b64encode(file.read()).decode('utf-8')
    return None

def get_weather_data():
    """Fetch real weather data from OpenWeatherMap API"""
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        # Fallback to simulated data
        return {
            'temperature': np.random.normal(28, 3),
            'humidity': np.random.normal(75, 10),
            'rainfall': np.random.exponential(2),
            'pressure': np.random.normal(1013, 5),
            'wind_speed': np.random.normal(15, 5),
            'description': 'Simulated weather'
        }
    
    try:
        # Use Tagudin, Ilocos Sur coordinates (16.9356° N, 120.4464° E)
        url = f"http://api.openweathermap.org/data/2.5/weather?lat=16.9356&lon=120.4464&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0),
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description']
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    # Fallback to simulated data
    return {
        'temperature': np.random.normal(28, 3),
        'humidity': np.random.normal(75, 10),
        'rainfall': np.random.exponential(2),
        'pressure': np.random.normal(1013, 5),
        'wind_speed': np.random.normal(15, 5),
        'description': 'Simulated weather'
    }

def send_email_alert(to_email, subject, message):
    """Send email alert to users"""
    if not EMAIL_AVAILABLE:
        print(f"Email alert simulation: To: {to_email}, Subject: {subject}")
        return True
    
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', '587'))
    smtp_username = os.environ.get('SMTP_USERNAME')
    smtp_password = os.environ.get('SMTP_PASSWORD')
    
    if not smtp_username or not smtp_password:
        print("SMTP credentials not configured - simulating email send")
        print(f"Email alert simulation: To: {to_email}, Subject: {subject}")
        return True
    
    try:
        msg = MimeMultipart()
        msg['From'] = smtp_username
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MimeText(message, 'html'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_username, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Email send error: {e}")
        return False

def send_emergency_alerts(alert):
    """Send emergency alerts to all users in affected areas"""
    def send_alerts_thread():
        users = User.query.filter_by(email_alerts=True).all()
        
        for user in users:
            if alert.affected_barangays:
                affected = json.loads(alert.affected_barangays)
                if user.barangay not in affected:
                    continue
            
            subject = f"🚨 EMERGENCY ALERT: {alert.title}"
            message = f"""
            <html>
            <body>
                <h2 style="color: red;">EMERGENCY ALERT - {alert.severity.upper()}</h2>
                <h3>{alert.title}</h3>
                <p>{alert.message}</p>
                <p><strong>Severity:</strong> {alert.severity.title()}</p>
                <p><strong>Alert Type:</strong> {alert.alert_type.replace('_', ' ').title()}</p>
                <p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p style="color: #666; font-size: 12px;">This is an automated message from EcoTrack Emergency System</p>
            </body>
            </html>
            """
            
            send_email_alert(user.email, subject, message)
    
    threading.Thread(target=send_alerts_thread).start()

def generate_password_reset_token():
    """Generate a secure random token for password reset"""
    return secrets.token_urlsafe(32)

def send_password_reset_email(user, token):
    """Send password reset email to user via Gmail"""
    reset_url = url_for('reset_password', token=token, _external=True)
    
    subject = "🔐 Password Reset Request - EcoTrack"
    message = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #1976d2; margin: 0;">EcoTrack</h1>
                <p style="color: #666; margin: 5px 0;">Flood Monitoring System</p>
            </div>
            
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                <h2 style="color: #1976d2; margin-top: 0;">Password Reset Request</h2>
                <p>Hello <strong>{user.first_name} {user.last_name}</strong>,</p>
                <p>You have requested to reset your password for your EcoTrack account.</p>
                <p>Click the button below to reset your password:</p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="{reset_url}" style="background-color: #1976d2; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Reset My Password</a>
            </div>
            
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p style="margin: 0; color: #856404;"><strong>⚠️ Important:</strong> This link will expire in 1 hour for security reasons.</p>
            </div>
            
            <p>If the button doesn't work, copy and paste this link into your browser:</p>
            <p style="word-break: break-all; color: #666; background-color: #f8f9fa; padding: 10px; border-radius: 4px;">{reset_url}</p>
            
            <div style="border-top: 1px solid #dee2e6; margin-top: 30px; padding-top: 20px;">
                <p style="color: #666; font-size: 14px; margin: 0;">If you didn't request this password reset, please ignore this email. Your account remains secure.</p>
                <p style="color: #666; font-size: 12px; margin: 10px 0 0 0;">This is an automated message from EcoTrack System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return send_email_alert(user.email, subject, message)

def calculate_flood_risk_score(report_data, weather_data):
    """Calculate enhanced AI-based flood risk score"""
    base_score = 0
    
    # Flood level multiplier (40% of total score)
    level_multipliers = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
    base_score += level_multipliers.get(report_data.get('flood_level', 'low'), 0.25) * 40
    
    # Weather conditions (30% of total score)
    if weather_data['rainfall'] > 50:
        base_score += 30
    elif weather_data['rainfall'] > 20:
        base_score += 20
    elif weather_data['rainfall'] > 10:
        base_score += 10
    
    # Wind speed impact
    if weather_data['wind_speed'] > 40:
        base_score += 15
    elif weather_data['wind_speed'] > 25:
        base_score += 10
    elif weather_data['wind_speed'] > 15:
        base_score += 5
    
    # Humidity impact
    if weather_data['humidity'] > 90:
        base_score += 10
    elif weather_data['humidity'] > 80:
        base_score += 5
    
    # Recent reports in same barangay (20% of total score)
    recent_reports = FloodReport.query.filter(
        FloodReport.barangay == report_data.get('barangay'),
        FloodReport.created_at >= datetime.utcnow() - timedelta(days=7)
    ).count()
    
    if recent_reports > 5:
        base_score += 20
    elif recent_reports > 3:
        base_score += 15
    elif recent_reports > 1:
        base_score += 10
    
    # Time of day factor (floods at night are more dangerous)
    current_hour = datetime.now().hour
    if 22 <= current_hour or current_hour <= 6:  # Night time
        base_score += 5
    
    # Weather condition factor
    weather_condition = report_data.get('weather_condition', 'sunny')
    if weather_condition == 'stormy':
        base_score += 15
    elif weather_condition == 'rainy':
        base_score += 10
    elif weather_condition == 'cloudy':
        base_score += 5
    
    return min(base_score, 100)

def get_flood_action_suggestions(flood_level, risk_score):
    """Get action suggestions based on flood level and risk score"""
    suggestions = FloodActionSuggestion.query.filter(
        FloodActionSuggestion.flood_level == flood_level,
        FloodActionSuggestion.risk_score_min <= risk_score,
        FloodActionSuggestion.risk_score_max >= risk_score,
        FloodActionSuggestion.is_active == True
    ).first()
    
    if not suggestions:
        # Fallback to closest match
        suggestions = FloodActionSuggestion.query.filter(
            FloodActionSuggestion.flood_level == flood_level,
            FloodActionSuggestion.is_active == True
        ).order_by(FloodActionSuggestion.risk_score_min).first()
    
    if suggestions:
        return {
            'title': suggestions.title,
            'description': suggestions.description,
            'actions': json.loads(suggestions.actions),
            'priority': suggestions.priority
        }
    
    return None

def send_automatic_flood_alert(report, risk_score, suggestions):
    """Send automatic flood alert based on risk assessment"""
    if risk_score >= 70:  # High or critical risk
        # Create emergency alert
        alert = EmergencyAlert(
            alert_type='flood_warning',
            severity='critical' if risk_score >= 80 else 'high',
            title=f'🚨 AUTOMATIC FLOOD ALERT - {report.barangay}',
            message=f'High-risk flood detected at {report.location}. Risk Score: {risk_score:.1f}/100. {suggestions["description"] if suggestions else "Please take immediate precautions."}',
            affected_barangays=json.dumps([report.barangay]),
            latitude=report.latitude,
            longitude=report.longitude,
            radius=5.0,
            created_by=report.user_id,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        db.session.add(alert)
        db.session.commit()
        
        # Send emergency alerts
        send_emergency_alerts(alert)
        
        return True
    return False

def convert_multiple_images_to_base64(files):
    """Convert multiple uploaded files to base64 JSON array"""
    if not files or files == [None]:
        return None
    
    images = []
    for file in files[:5]:
        if file and file.filename:
            try:
                image_data = base64.b64encode(file.read()).decode('utf-8')
                images.append({
                    'filename': secure_filename(file.filename),
                    'data': image_data
                })
            except Exception as e:
                print(f"Error processing image: {e}")
    
    return json.dumps(images) if images else None

def generate_evacuation_map(barangay):
    """Generate evacuation map for a specific barangay"""
    safe_zones = SafeZone.query.filter_by(barangay=barangay, is_active=True).all()
    routes = EvacuationRoute.query.filter_by(from_barangay=barangay, is_active=True).all()
    
    # Center on Tagudin, Ilocos Sur
    map_center = [16.9356, 120.4464]
    m = folium.Map(location=map_center, zoom_start=14)
    
    for zone in safe_zones:
        folium.Marker(
            [zone.latitude, zone.longitude],
            popup=f"<b>{zone.name}</b><br>Capacity: {zone.capacity}<br>{zone.address}",
            tooltip=zone.name,
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)
    
    for route in routes:
        if route.route_points:
            points = json.loads(route.route_points)
            folium.PolyLine(
                points,
                weight=5,
                color='blue',
                opacity=0.7,
                popup=f"Route: {route.name}<br>Distance: {route.distance_km}km<br>Est. Time: {route.estimated_time}min"
            ).add_to(m)
    
    return m._repr_html_()

def predict_flood_risk():
    """Use machine learning to predict flood risk"""
    reports = FloodReport.query.all()
    if len(reports) < 10:
        return "Insufficient data for prediction"
    
    data = []
    for report in reports:
        weather = get_weather_data()
        data.append([
            weather['rainfall'],
            weather['humidity'],
            weather['pressure'],
            weather['wind_speed'],
            1 if report.flood_level in ['high', 'critical'] else 0
        ])
    
    df = pd.DataFrame(data, columns=['rainfall', 'humidity', 'pressure', 'wind_speed', 'flood_risk'])
    
    if df['flood_risk'].sum() == 0:
        return "Low Risk"
    
    X = df[['rainfall', 'humidity', 'pressure', 'wind_speed']]
    y = df['flood_risk']
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        current_weather = get_weather_data()
        prediction = model.predict([[
            current_weather['rainfall'],
            current_weather['humidity'],
            current_weather['pressure'],
            current_weather['wind_speed']
        ]])
        
        risk_level = "High Risk" if prediction[0] > 0.5 else "Low Risk"
        return risk_level
    except:
        return "Moderate Risk"

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            recent_reports = FloodReport.query.order_by(FloodReport.created_at.desc()).limit(5).all()
        else:
            recent_reports = FloodReport.query.filter_by(user_id=current_user.id).order_by(FloodReport.created_at.desc()).limit(5).all()
        flood_prediction = predict_flood_risk()
        return render_template('dashboard.html', reports=recent_reports, prediction=flood_prediction)
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            # If email_alerts is used as active flag and set to False, block login
            if user.email_alerts is False:
                flash('Your account is deactivated. Please contact an administrator.', 'error')
                return render_template('login.html', form=form)
            login_user(user, remember=True)
            user.last_login = datetime.utcnow()
            session['user_id'] = user.id
            session['user_role'] = user.role
            session['last_activity'] = datetime.now().isoformat()
            session.permanent = True
            db.session.commit()
            flash(f'Welcome back, {user.first_name}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please check your credentials and try again.', 'error')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists')
            return render_template('register.html', form=form)
        
        existing_email = User.query.filter_by(email=form.email.data).first()
        if existing_email:
            flash('Email already registered')
            return render_template('register.html', form=form)
        
        user = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data, method='pbkdf2:sha256'),
            role=form.role.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            barangay=form.barangay.data,
            phone=form.phone.data
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    # Get user info before logout for logging
    user_info = f"{current_user.username} ({current_user.role})" if current_user.is_authenticated else "Unknown"
    
    # Clear Flask-Login session
    logout_user()
    
    # Clear all session data
    session.clear()
    
    # Create response with security headers
    response = redirect(url_for('login'))
    
    # Add comprehensive security headers to prevent caching and back button access
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = '0'
    
    # Clear any authentication-related cookies
    response.set_cookie('remember_token', '', expires=0)
    response.set_cookie('session', '', expires=0)
    response.set_cookie('user_id', '', expires=0)
    response.set_cookie('user_role', '', expires=0)
    
    flash('You have been logged out successfully. All sessions have been cleared.')
    print(f"User {user_info} logged out successfully")
    
    return response

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # Generate token and save to database
            token = generate_password_reset_token()
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            # Delete any existing tokens for this user
            PasswordResetToken.query.filter_by(user_id=user.id).delete()
            
            reset_token = PasswordResetToken(
                user_id=user.id,
                token=token,
                expires_at=expires_at
            )
            db.session.add(reset_token)
            db.session.commit()
            
            # Send email
            if send_password_reset_email(user, token):
                flash('Password reset link has been sent to your email. Please check your inbox and spam folder.', 'success')
                print(f"Password reset email sent to {user.email}")
            else:
                flash('Error sending email. Please check your Gmail configuration or contact support.', 'error')
        else:
            # Don't reveal if email exists or not for security
            flash('If an account with that email exists, a password reset link has been sent.', 'info')
        
        return redirect(url_for('login'))
    
    return render_template('forgot_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    # Verify token
    reset_token = PasswordResetToken.query.filter_by(token=token).first()
    if not reset_token or reset_token.expires_at < datetime.utcnow():
        flash('Invalid or expired password reset link. Please request a new one.')
        return redirect(url_for('forgot_password'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        # Update password
        user = reset_token.user
        user.password_hash = generate_password_hash(form.new_password.data, method='pbkdf2:sha256')
        
        # Delete the used token
        db.session.delete(reset_token)
        db.session.commit()
        
        flash('Your password has been reset successfully. You can now login with your new password.')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', form=form, token=token)

@app.route('/report_flood', methods=['GET', 'POST'])
@login_required
def report_flood():
    if current_user.role not in ['resident', 'brgy_official']:
        flash('Access denied')
        return redirect(url_for('index'))
    
    form = FloodReportForm()
    # Pre-fill and enforce barangay to the user's registered barangay
    if request.method == 'GET' and current_user.barangay:
        form.barangay.data = current_user.barangay
    elif request.method == 'POST' and current_user.barangay:
        # Override any submitted value to ensure it matches the user's permanent barangay
        form.barangay.data = current_user.barangay
    if form.validate_on_submit():
        weather_data = get_weather_data()
        flood_images_data = convert_multiple_images_to_base64(form.flood_images.data)
        latitude = float(form.latitude.data) if form.latitude.data else None
        longitude = float(form.longitude.data) if form.longitude.data else None
        
        report_data = {
            'flood_level': form.flood_level.data,
            'barangay': form.barangay.data,
            'weather_condition': form.weather_condition.data
        }
        
        risk_score = calculate_flood_risk_score(report_data, weather_data)
        
        report = FloodReport(
            user_id=current_user.id,
            location=form.location.data,
            barangay=form.barangay.data,
            latitude=latitude,
            longitude=longitude,
            flood_level=form.flood_level.data,
            description=form.description.data,
            weather_condition=form.weather_condition.data,
            flood_images=flood_images_data,
            risk_score=risk_score,
            evacuation_needed=(form.flood_level.data in ['high', 'critical'] and risk_score > 70)
        )
        db.session.add(report)
        db.session.commit()
        
        # Get action suggestions based on risk assessment
        suggestions = get_flood_action_suggestions(form.flood_level.data, risk_score)
        
        # Send automatic flood alert if high risk
        alert_sent = send_automatic_flood_alert(report, risk_score, suggestions)
        
        # Prepare response message with suggestions
        if suggestions:
            flash(f'Flood report submitted successfully! Risk Score: {risk_score:.1f}/100. {suggestions["title"]}', 'success')
            # Store suggestions in session for display
            session['flood_suggestions'] = suggestions
        else:
            flash(f'Flood report submitted successfully! Risk Score: {risk_score:.1f}/100', 'success')
        
        if alert_sent:
            flash('🚨 High-risk flood detected! Emergency alert has been sent to all users in the area.', 'warning')
        
        return redirect(url_for('flood_report_success', report_id=report.id))
    return render_template('report_flood.html', form=form)

@app.route('/flood_report_success/<int:report_id>')
@login_required
def flood_report_success(report_id):
    """Show flood report success page with action suggestions"""
    report = FloodReport.query.get_or_404(report_id)
    suggestions = session.pop('flood_suggestions', None)
    
    # Get nearby safe zones
    safe_zones = []
    if report.latitude and report.longitude:
        safe_zones = SafeZone.query.filter_by(barangay=report.barangay, is_active=True).all()
    
    return render_template('flood_report_success.html', 
                         report=report, 
                         suggestions=suggestions,
                         safe_zones=safe_zones)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm()
    change_form = ChangePasswordForm()
    if form.validate_on_submit():
        if form.profile_image.data:
            current_user.profile_image = convert_image_to_base64(form.profile_image.data)
        
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        current_user.email = form.email.data
        current_user.barangay = form.barangay.data
        current_user.phone = form.phone.data
        db.session.commit()
        flash('Profile updated successfully')
        return redirect(url_for('profile'))
    elif request.method == 'GET':
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
        form.barangay.data = current_user.barangay
        form.phone.data = current_user.phone
    return render_template('profile.html', form=form, change_form=change_form)

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    change_form = ChangePasswordForm()
    if change_form.validate_on_submit():
        if not check_password_hash(current_user.password_hash, change_form.current_password.data):
            flash('Current password is incorrect')
            return redirect(url_for('profile'))
        current_user.password_hash = generate_password_hash(change_form.new_password.data, method='pbkdf2:sha256')
        db.session.commit()
        flash('Password updated successfully')
        return redirect(url_for('profile'))
    # Show first validation error if any
    for field_name, errors in change_form.errors.items():
        if errors:
            flash(errors[0])
            break
    return redirect(url_for('profile'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    total_reports = FloodReport.query.count()
    total_users = User.query.count()
    pending_reports = FloodReport.query.filter_by(status='pending').count()
    
    recent_reports = FloodReport.query.order_by(FloodReport.created_at.desc()).limit(10).all()
    recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    # Load recent feedback for admin to review (latest 10)
    feedback_items = Feedback.query.order_by(Feedback.created_at.desc()).limit(10).all()

    return render_template('admin_dashboard.html', 
                         total_reports=total_reports,
                         total_users=total_users,
                         pending_reports=pending_reports,
                         recent_reports=recent_reports,
                         recent_users=recent_users,
                         feedback_items=feedback_items)

@app.route('/admin/analytics')
@login_required
def analytics():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    barangay = request.args.get('barangay')
    flood_level = request.args.get('flood_level')
    
    query = FloodReport.query
    
    if date_from:
        query = query.filter(FloodReport.created_at >= datetime.strptime(date_from, '%Y-%m-%d'))
    if date_to:
        query = query.filter(FloodReport.created_at <= datetime.strptime(date_to, '%Y-%m-%d'))
    if barangay:
        query = query.filter(FloodReport.barangay.ilike(f'%{barangay}%'))
    if flood_level:
        query = query.filter(FloodReport.flood_level == flood_level)
    
    reports = query.all()
    
    flood_levels = [r.flood_level for r in reports]
    barangays = [r.barangay for r in reports]
    dates = [r.created_at.date() for r in reports]
    
    flood_level_counts = pd.Series(flood_levels).value_counts()
    barangay_counts = pd.Series(barangays).value_counts()
    date_counts = pd.Series(dates).value_counts().sort_index()
    
    charts = {
        'flood_levels': {
            'labels': flood_level_counts.index.tolist(),
            'values': flood_level_counts.values.tolist()
        },
        'barangays': {
            'labels': barangay_counts.head(10).index.tolist(),
            'values': barangay_counts.head(10).values.tolist()
        },
        'timeline': {
            'dates': [str(d) for d in date_counts.index.tolist()],
            'counts': date_counts.values.tolist()
        }
    }
    
    return render_template('analytics.html', charts=charts, reports=reports)

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/<int:user_id>/update', methods=['POST'])
@login_required
def admin_update_user(user_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    user = User.query.get_or_404(user_id)
    data = request.get_json(silent=True) or {}
    for field in ['email', 'role', 'barangay', 'phone', 'email_alerts']:
        if field in data:
            setattr(user, field, data[field])
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    user = User.query.get_or_404(user_id)
    # delete user's reports first
    for r in FloodReport.query.filter_by(user_id=user.id).all():
        db.session.delete(r)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/reports')
@login_required
def admin_reports():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    barangay = request.args.get('barangay')
    status = request.args.get('status')
    flood_level = request.args.get('flood_level')

    query = FloodReport.query.order_by(FloodReport.created_at.desc())
    if barangay:
        query = query.filter(FloodReport.barangay == barangay)
    if status:
        query = query.filter(FloodReport.status == status)
    if flood_level:
        query = query.filter(FloodReport.flood_level == flood_level)

    reports = query.all()
    return render_template('admin_reports.html', reports=reports)

@app.route('/admin/report/<int:report_id>/update_status', methods=['POST'])
@login_required
def update_report_status(report_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    status = request.json.get('status')
    
    report = FloodReport.query.get_or_404(report_id)
    report.status = status
    report.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/admin/users/<int:user_id>/update', methods=['POST'])
@login_required
def update_user(user_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    user = User.query.get_or_404(user_id)
    data = request.get_json(silent=True) or {}
    # Update allowed fields if present
    for field in ['first_name', 'last_name', 'email', 'role', 'barangay', 'phone']:
        if field in data and data[field] is not None:
            setattr(user, field, data[field])
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    user = User.query.get_or_404(user_id)
    # Delete user's reports first to satisfy FK
    user_reports = FloodReport.query.filter_by(user_id=user.id).all()
    for r in user_reports:
        db.session.delete(r)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/admin/report/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_report(report_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    report = FloodReport.query.get_or_404(report_id)
    db.session.delete(report)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Empty'}), 400
    fb = Feedback(user_id=current_user.id, text=text)
    db.session.add(fb)
    db.session.commit()
    return jsonify({'success': True})

@app.after_request
def add_no_cache_headers(response):
    """Prevent caching so protected pages aren't accessible after logout via back/forward cache."""
    # Only apply to HTML responses (not static files)
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, private'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Last-Modified'] = '0'
        
        # Additional security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
    
    return response

@app.route('/admin/emergency_alert', methods=['GET', 'POST'])
@login_required
def emergency_alert():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    form = EmergencyAlertForm()
    if form.validate_on_submit():
        affected_barangays = []
        if form.affected_barangays.data:
            affected_barangays = [b for b in form.affected_barangays.data if b]
        
        alert = EmergencyAlert(
            alert_type=form.alert_type.data,
            severity=form.severity.data,
            title=form.title.data,
            message=form.message.data,
            affected_barangays=json.dumps(affected_barangays),
            latitude=float(form.latitude.data) if form.latitude.data else None,
            longitude=float(form.longitude.data) if form.longitude.data else None,
            radius=form.radius.data,
            created_by=current_user.id,
            expires_at=form.expires_at.data
        )
        db.session.add(alert)
        db.session.commit()
        
        send_emergency_alerts(alert)
        
        flash('Emergency alert sent successfully')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('emergency_alert.html', form=form)

@app.route('/admin/safe_zones', methods=['GET', 'POST'])
@login_required
def manage_safe_zones():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    form = SafeZoneForm()
    if form.validate_on_submit():
        facilities = []
        if form.facilities.data:
            facilities = [f.strip() for f in form.facilities.data.split('\n') if f.strip()]
        
        safe_zone = SafeZone(
            name=form.name.data,
            address=form.address.data,
            latitude=float(form.latitude.data),
            longitude=float(form.longitude.data),
            capacity=int(form.capacity.data),
            barangay=form.barangay.data,
            contact_person=form.contact_person.data,
            contact_phone=form.contact_phone.data,
            facilities=json.dumps(facilities)
        )
        db.session.add(safe_zone)
        db.session.commit()
        
        flash('Safe zone added successfully')
        return redirect(url_for('manage_safe_zones'))
    
    safe_zones = SafeZone.query.order_by(SafeZone.created_at.desc()).all()
    return render_template('safe_zones.html', form=form, safe_zones=safe_zones)

@app.route('/flood_map')
@login_required
def flood_map():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    reports = FloodReport.query.filter(
        FloodReport.latitude.isnot(None),
        FloodReport.longitude.isnot(None),
        FloodReport.status.in_(['pending', 'verified'])
    ).all()
    
    safe_zones = SafeZone.query.filter_by(is_active=True).all()
    
    # Initialize map; we'll fit to report bounds if available
    map_center = [16.9356, 120.4464]  # Default center on Tagudin, Ilocos Sur
    m = folium.Map(location=map_center, zoom_start=14)
    
    colors = {'low': 'green', 'medium': 'orange', 'high': 'red', 'critical': 'darkred'}
    
    bounds = []
    for report in reports:
        color = colors.get(report.flood_level, 'blue')
        risk_score_text = f"{report.risk_score:.1f}" if report.risk_score is not None else "N/A"
        image_html = ""
        if report.flood_images:
            try:
                images = json.loads(report.flood_images)
                if images and isinstance(images, list) and images[0].get('data'):
                    image_html = f"<br><img src=\"data:image/jpeg;base64,{images[0]['data']}\" style=\"width:220px;height:auto;border-radius:4px;margin-top:6px;\" />"
            except Exception:
                pass
        # Ensure coordinates are plausible; auto-correct if lat/lon were swapped
        marker_lat = float(report.latitude)
        marker_lon = float(report.longitude)
        in_ph = (4.0 <= marker_lat <= 21.5) and (116.0 <= marker_lon <= 127.5)
        if not in_ph:
            swapped_in_ph = (4.0 <= marker_lon <= 21.5) and (116.0 <= marker_lat <= 127.5)
            if swapped_in_ph:
                marker_lat, marker_lon = marker_lon, marker_lat
        # Skip clearly invalid points
        if not ((-90.0 <= marker_lat <= 90.0) and (-180.0 <= marker_lon <= 180.0)):
            continue
        popup_text = (
            f"<b>{report.location}</b><br>"
            f"Level: {report.flood_level.title()}<br>"
            f"Reporter: {report.reporter.first_name} {report.reporter.last_name}<br>"
            f"Time: {report.created_at.strftime('%Y-%m-%d %H:%M')}<br>"
            f"Risk Score: {risk_score_text}/100<br>"
            f"{report.description[:120]}..."
            f"{image_html}"
        )
        
        folium.CircleMarker(
            [marker_lat, marker_lon],
            radius=8 + (2 * (['low', 'medium', 'high', 'critical'].index(report.flood_level))),
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)
        bounds.append([marker_lat, marker_lon])

    if bounds:
        try:
            m.fit_bounds(bounds)
        except Exception:
            pass
    
    for zone in safe_zones:
        folium.Marker(
            [zone.latitude, zone.longitude],
            popup=f"<b>SAFE ZONE: {zone.name}</b><br>Capacity: {zone.capacity}<br>{zone.address}",
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(m)
    
    map_html = m._repr_html_()
    return render_template('flood_map.html', map_html=map_html)

@app.route('/evacuation/<barangay>')
@login_required
def evacuation_routes(barangay):
    map_html = generate_evacuation_map(barangay)
    return render_template('evacuation_routes.html', barangay=barangay, map_html=map_html)

@app.route('/export_reports')
@login_required
def export_reports():
    if current_user.role != 'admin':
        flash('Admin access required')
        return redirect(url_for('index'))
    
    reports = FloodReport.query.order_by(FloodReport.created_at.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'ID', 'Reporter', 'Location', 'Barangay', 'Flood Level', 'Description',
        'Weather Condition', 'Latitude', 'Longitude', 'Risk Score',
        'Status', 'Created At', 'Updated At'
    ])
    
    for report in reports:
        writer.writerow([
            report.id,
            f"{report.reporter.first_name} {report.reporter.last_name}",
            report.location,
            report.barangay,
            report.flood_level,
            report.description,
            report.weather_condition or '',
            report.latitude or '',
            report.longitude or '',
            report.risk_score or '',
            report.status,
            report.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            report.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    output.seek(0)
    
    return app.response_class(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=flood_reports.csv"}
    )

@app.route('/api/weather')
def api_weather():
    weather = get_weather_data()
    return jsonify(weather)

@app.route('/api/alerts')
@login_required
def api_alerts():
    # Fetch active alerts
    active_alerts = EmergencyAlert.query.filter_by(is_active=True).filter(
        EmergencyAlert.expires_at > datetime.utcnow()
    ).order_by(EmergencyAlert.created_at.desc()).all()

    # Personalize by barangay if specified
    user_barangay = (current_user.barangay or '').strip()

    def is_relevant(alert: EmergencyAlert) -> bool:
        if not alert.affected_barangays:
            return True
        try:
            affected = json.loads(alert.affected_barangays)
            if isinstance(affected, list) and user_barangay:
                return user_barangay in affected
        except Exception:
            pass
        # If parsing fails, show to all
        return True

    alerts_data = []
    for alert in active_alerts:
        if not is_relevant(alert):
            continue
        alerts_data.append({
            'id': alert.id,
            'type': alert.alert_type,
            'severity': alert.severity,
            'title': alert.title,
            'message': alert.message,
            'created_at': alert.created_at.isoformat(),
            'expires_at': alert.expires_at.isoformat() if alert.expires_at else None
        })

    return jsonify(alerts_data)

@app.route('/api/flood-suggestions/<flood_level>/<int:risk_score>')
def api_flood_suggestions(flood_level, risk_score):
    """API endpoint to get flood action suggestions"""
    suggestions = get_flood_action_suggestions(flood_level, risk_score)
    if suggestions:
        return jsonify(suggestions)
    else:
        return jsonify({'error': 'No suggestions found'}), 404

@app.route('/api/notifications')
@login_required
def api_notifications():
    """Get notifications for the current user"""
    notifications = []
    
    # Get emergency alerts
    alerts = EmergencyAlert.query.filter_by(is_active=True).filter(
        EmergencyAlert.expires_at > datetime.utcnow()
    ).order_by(EmergencyAlert.created_at.desc()).limit(10).all()
    
    for alert in alerts:
        # Check if alert is relevant to user's barangay
        is_relevant = True
        if alert.affected_barangays and current_user.barangay:
            try:
                affected = json.loads(alert.affected_barangays)
                if isinstance(affected, list):
                    is_relevant = current_user.barangay in affected
            except Exception:
                pass
        
        if is_relevant:
            notifications.append({
                'id': f'alert_{alert.id}',
                'type': 'emergency_alert',
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity,
                'created_at': alert.created_at.isoformat(),
                'is_read': False
            })
    
    # Get flood reports for user's barangay (if they're a resident/brgy official)
    if current_user.role in ['resident', 'brgy_official'] and current_user.barangay:
        reports = FloodReport.query.filter_by(
            barangay=current_user.barangay,
            status='pending'
        ).order_by(FloodReport.created_at.desc()).limit(5).all()
        
        for report in reports:
            notifications.append({
                'id': f'report_{report.id}',
                'type': 'admin_notification',
                'title': f'New Flood Report - {report.location}',
                'message': f'Flood level: {report.flood_level.title()}. {report.description[:100]}...',
                'severity': report.flood_level,
                'created_at': report.created_at.isoformat(),
                'is_read': False
            })
    
    # Get system notifications for admins
    if current_user.role == 'admin':
        # Pending reports count
        pending_count = FloodReport.query.filter_by(status='pending').count()
        if pending_count > 0:
            notifications.append({
                'id': 'admin_pending_reports',
                'type': 'admin_notification',
                'title': f'{pending_count} Pending Reports',
                'message': f'You have {pending_count} flood reports awaiting review.',
                'severity': 'medium',
                'created_at': datetime.utcnow().isoformat(),
                'is_read': False
            })
        
        # New user registrations (last 24 hours)
        new_users = User.query.filter(
            User.created_at >= datetime.utcnow() - timedelta(days=1),
            User.role != 'admin'
        ).count()
        if new_users > 0:
            notifications.append({
                'id': 'admin_new_users',
                'type': 'admin_notification',
                'title': f'{new_users} New Users',
                'message': f'{new_users} new users registered in the last 24 hours.',
                'severity': 'low',
                'created_at': datetime.utcnow().isoformat(),
                'is_read': False
            })
    
    return jsonify(notifications)

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            create_admin_user()
            create_default_action_suggestions()
            print("Database initialized successfully!")
        except Exception as e:
            print(f"Database initialization error: {e}")
            db.drop_all()
            db.create_all()
            create_admin_user()
            create_default_action_suggestions()
            print("Database recreated successfully!")
    app.run(host='0.0.0.0', port=5000, debug=True)
