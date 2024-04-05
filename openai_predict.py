# Install all dependencies required below.
# !pip install pandas numpy seaborn matplotlib scikit-learn neattext
import openai
import pandas as pd
import ast 

# Load EDA Pkgs
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

 # Load Dataset
df = pd.read_csv("/Users/stephen/Desktop/SPROUT/[SPROUT] CA Project 2.0/database/[SPROUT] 03 Sample External Dataset - Sheet1.csv")
df = df.rename(columns=lambda x: x.strip())

# Concatenate Ticket Subject and Body
df['Complete Ticket'] = df['Client Complaint'].str.cat(df['Ticket Body'], sep='; ')
df.insert(loc=2, column='Complete Ticket', value=df.pop('Complete Ticket'))

# Turn int into a str
df['Support Level'] = df['Support Level'].astype(str)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# OpenAI Key
openai.api_key = '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'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

# Load Dataset
df = pd.read_csv("/Users/stephen/Desktop/SPROUT/[SPROUT] CA Project 2.0/database/[SPROUT] 03 Sample External Dataset - Sheet1.csv")
product = df['Type of Product'].unique()
priority = df['Priority'].unique()
type_complaint = df['Type of Complaint'].unique()
support = df['Support Level'].unique()

# Route to display the HTML form
@app.route('/')
def index():
    return render_template('submit_page.html')

# Route to handle the form submission; initial HTML
@app.route('/submit', methods=['POST'])
def submit():
    # Retrieving data from form
    subject_type = request.form['subjectType']
    ticket_body = request.form['ticketBody']
    
    global input_text
    input_text = subject_type + "; " + ticket_body

    prompt = f"""
    Look at the information from '''{df}. Understand the relationships between the
    columns, rows, and how the values are connected to each other. These are customer
    complaints that are categorized into certain tags, specificially: Type of Product, Priority,
    Type of Complaint, and Support Level. Identify why, and train yourself on the distinctions.
    Train yourself 3 times.

    Now, I want you to predict the 4 different labels for a new set of client complaint.
    Please provide the tags of the following text (concatenated):
    '''{input_text}'''.

    After summarizing and analyzing the text, please classify 
    the ticket with the following labels:

    - Type of Product: 
    - Priority: 
    - Type of Complaint: 
    - Support Level:

    The FINAL and ONLY output should be in a Python list format, from product, priority, complaint, and support.

    These must all be based from the list of '''{product}''', '''{priority}''', '''{type_complaint}''', and '''{support}'''.
    """
    
    ticket_output = get_completion(prompt)
    tag_list = ast.literal_eval(ticket_output)
    print(tag_list)
    print(type(tag_list))

    global value_1, value_2, value_3, value_4
    value_1 = tag_list[0]
    value_2 = tag_list[1]
    value_3 = tag_list[2]
    value_4 = tag_list[3]

    product_crit = f"""
    Mobile Payments: GCash allows users to make mobile payments securely and conveniently.
    GSave: A high-yield savings account that offers users options between CIMB Bank Philippines, BPI, and Maybank Philippines.
    GCredit: A revolving mobile credit line initially powered by Fuse Lending, later transferred to CIMB Bank.
    GCash Padala: A remittance service available to both app users and non-app users through partner outlets.
    GCash Jr.: Designed for users aged 7 to 17, offering a tailored experience.
    Double Safe: A security feature requiring facial identification from customers to enhance safety.
    GForest: Allows users to collect green energy by engaging in eco-friendly activities.
    GLife: An app where users can shop for various products from their favorite brands.
    KKB: Enables users to split bills with friends, even if they do not have GCash.
    GGives: Offers a buy now, pay later service with flexible payments.
    GInsure: Provides affordable insurance options within the GCash app.
    GCash Pera Outlet: Allows individuals to earn money by becoming a GCash Pera Outlet.
    GCredit: Provides users with a credit line for extended budget flexibility.
    GLoan: Offers pre-approved access to cash loans instantly without collateral.
    GFunds: Enables users to invest in funds managed by partner providers.
    GSave: A feature that helps users save for the future conveniently within the GCash app.
    """

    priority_crit = f"""
    Urgent: Issues causing critical service disruption or financial loss.
    Not Urgent: Non-critical issues with no immediate impact on operations or customer satisfaction.
    Normal: Routine inquiries or requests not requiring immediate attention or action.
    """

    complaint_crit = f"""
    Account Issue: Customer account-related problems with significant impact.
    Transaction Failure: Failures in financial transactions with substantial consequences.
    Technical Issue: Technical problems affecting the service or application.
    Claim Issue: Disputes or problems related to claims processing.
    Account Access: Difficulties accessing the customer's account or system.
    Billing Error: Errors in billing or invoicing processes.
    Application Error: Errors or malfunctions within the application or software.
    Payment Issue: Issues related to payment processing or transactions.
    Disbursement Issue: Problems with disbursement or distribution of funds.
    Service Issue: General issues with the service provided.
    Activation Error: Errors during the activation process of a service or product.
    """

    support_crit = f"""
    1 - Requires extensive specialized support and engineering assistance.
    2 - Requires minor specialized support from engineering or technical staff.
    3 - Requires assistance from experienced Customer Advocacy (CA) members.
    4 - Can be handled by any member of Customer Advocacy (CA) team.
    """

    prompt2 = f"""
    Now, based on the criteria presented below, explain the reason for 
    such given tags in '''{tag_list}'''. The context for your explanation
    will come from '''{input_text}'''. You will need to explain why each label is given their 
    corresponding values, and not just reiterate the values themselves.

    Explanation Format: Ticket was given product tag 'GCredit' because it primarily revolves around mobile credit concerns raised.
    Per explanation, do not include any commas (,). The commas can only separate the values from each other.
    Be straightforward.

    Here are the criteria as basis for your explanation:
    '''{product_crit}'''
    '''{priority_crit}'''
    '''{complaint_crit}'''
    '''{support_crit}'''

    The FINAL and ONLY output should be in a Python LIST format, containing the explanation to the product type,
    the explanation to the priority, explanation to the complaint type, and explanation to support level.

    List Format: ['Explanation to Product Type', 'Explanation to Priority', 'Explanation to Complaint Type', 'Explanation to Support Level']
    Please print the list in 1 complete line.
    """

    rationale_output = get_completion(prompt2)
    rationale_list = rationale_output.strip('[]').split(',')

    # Trim whitespace from each element in the list
    final_rationale = [item.strip() for item in rationale_list]
    
    global rationale_1, rationale_2, rationale_3, rationale_4
    rationale_1 = final_rationale[0]
    rationale_2 = final_rationale[1]
    rationale_3 = final_rationale[2]
    rationale_4 = final_rationale[3]

    return redirect(url_for('table'))

# Final HTML
@app.route('/table')
def table():
    return render_template('results_page.html', 
                           type_product=value_1, 
                           priority=value_2, 
                           type_complaint=value_3, 
                           support=value_4,
                           rat_product=rationale_1,
                           rat_prio=rationale_2,
                           rat_complaint=rationale_3,
                           rat_support=rationale_4
                           )

if __name__ == '__main__':
    app.run(debug=True)