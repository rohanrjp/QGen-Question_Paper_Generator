from flask import render_template, url_for, flash, redirect, request,session
from questionpapergenerator import app, db
from PyPDF2 import PdfReader
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast
from flask_login import login_user, current_user, logout_user, login_required
from questionpapergenerator.forms import RegistrationForm, LoginForm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

'''def preprocess_text(text):
    # Remove leading and trailing whitespace
    text = text.strip()
    # Replace newlines and multiple whitespaces with a single space
    text = ' '.join(text.split())
    # Split the text into paragraphs
    paragraphs = text.split('\n')
    # Combine paragraphs into a single paragraph
    single_paragraph = ' '.join(paragraphs)
    return single_paragraph
'''

def preprocess_text(text):
    # Remove leading and trailing whitespace
    text = text.strip()

    # Replace bullet points with a space
    text = re.sub(r'\s*•\s*', ' ', text)

    # Replace newlines and multiple whitespaces with a single space
    text = ' '.join(text.split())

    # Split the text into paragraphs
    paragraphs = text.split('\n')

    # Combine paragraphs into a single paragraph
    single_paragraph = ' '.join(paragraphs)

    return single_paragraph


checkpoint = "t5-base"
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained("ThomasSimonini/t5-end2end-question-generation")

import random

def hf_run_model(input_string, num_return_sequences=8, num_questions=5, max_sequence_length=512, generator_args=None):
    if generator_args is None:
        generator_args = {
            "max_length": max_sequence_length,
            "num_beams": 10,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 6,
            "early_stopping": True,
        }
    input_string = "generate questions: " + input_string + " </s>"
    input_ids = tokenizer.encode(input_string, truncation=True, max_length=max_sequence_length, return_tensors="pt")

    generated_questions = []
    unique_questions = set()
    
    # Generate questions using the model
    res = model.generate(input_ids, **generator_args, num_return_sequences=num_return_sequences)
    output = tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    for sequence in output:
        sequence = sequence.split("<sep>")
        questions = [question.strip() for question in sequence[0].split("?") if question.strip()]
        generated_questions.extend(questions)
    
    # Randomly sample questions until reaching the desired number of non-repeated questions
    while len(unique_questions) < num_questions and generated_questions:
        question = random.choice(generated_questions)
        generated_questions.remove(question)
        if question not in unique_questions:
            unique_questions.add(question)
    
    return list(unique_questions)

def convert_list_to_pdf_with_template(data_list, output_file):
    # Create the PDF canvas
    c = canvas.Canvas(output_file, pagesize=letter)

    # Set the font and size
    c.setFont("Helvetica", 12)

    # Add the template or background image
    template_path = 'template.png'
    c.drawImage(template_path, 0, 0, width=letter[0], height=letter[1])

    # Write the list elements to the PDF
    y = 550  # Starting y position
    for item in data_list:
        c.drawString(100, y, str(item))
        y -= 50

    # Save the canvas as the final PDF
    c.save()



'''
@app.route("/login")
def login():
        form=LoginForm()
        return render_template('login.html',form=form)
'''

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)

@app.route("/")
@app.route("/userdashboard")
def user_dashboard():
    return render_template('dashboard.html')


""""
@app.route("/upload")
def upload():
    return render_template('upload.html')
"""

'''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = '/' + file.filename
            file.save(file_path)
            # Perform text extraction
            extracted_text = extract_text_from_pdf(file_path)
            # Render the extracted text
            preprocessed_text = preprocess_text(extracted_text)
            print(preprocessed_text)
            questions= hf_run_model(preprocessed_text)
            return render_template('result.html', text1 = extracted_text, text2=preprocessed_text, text3=questions)
    return render_template('upload.html')
    '''
import os

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file to a temporary directory
            temp_dir = '/tmp'
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            
            # Perform text extraction
            extracted_text = extract_text_from_pdf(file_path)
            
            # Delete the temporary file
            os.remove(file_path)
            
            # Continue with text processing
            preprocessed_text = preprocess_text(extracted_text)
            print(preprocessed_text)
            questions = hf_run_model(preprocessed_text,num_return_sequences=8, num_questions=5)
            session['questions'] = questions
            for count,ele in enumerate(questions):
                print(count+1)
                print(ele)
            return render_template('upload.html', text1=extracted_text, text2=preprocessed_text, text3=questions)
        else:
            session['questions'] = [] 
        return redirect(url_for('generate_pdf'))
    return render_template('upload.html')

@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    questions = session.get('questions')
    output_path='output.pdf'
    if questions:
        result = convert_list_to_pdf_with_template(questions,output_path)
        return result
    return "No questions found"



@app.route('/generate_pdf_endpoint', methods=['POST'])
def generate_pdf_endpoint():
    questions = session.get('questions')
    result = generate_pdf(questions)
    return result
