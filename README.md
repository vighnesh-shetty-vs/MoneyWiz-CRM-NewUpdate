MoneyWiz CRM is a Streamlit-based CRM and analytics platform designed for retail businesses. It bridges operational sales management and managerial insights with:

Salesperson Workspace: Customer CRUD operations, search, and personal sales analytics.
Region Manager Dashboard: KPIs, revenue forecasting, customer segmentation, and product recommendations.


Live Application
👉 https://moneywiz-crm-newupdate-bexgwqs8fhq96q5aks7anm.streamlit.app/

Features
Salesperson

Add, update, delete, and search customer records.
Auto-calculated pricing and timestamp-based OrderID.
Personal analytics: product-wise revenue charts.

Region Manager

Standard KPIs: Region, store, and salesperson performance.
Sales Forecast: 3-month prediction using Linear Regression.
Customer Segmentation: K-Means clustering into VIP, Growing Leads, Occasional Buyers.
Product Recommendations: Cross-sell suggestions via correlation analysis.


Tech Stack

ComponentTechnologyFrontendStreamlitBackend/DBSQLite + SQLAlchemyData ProcessingPandas, NumPyML ModelsScikit-learnVisualizationPlotly Express

Installation

Clone the repository:
Shellgit clone https://github.com/vighnesh-shetty-vs/MoneyWiz-CRM-NewUpdate.gitShow more lines

Install dependencies:
Shellpip install -r requirements.txtShow more lines

Place customers.xlsx in the root directory.
Configure database in .streamlit/secrets.toml.
Run the app:
Shellstreamlit run app.pyShow more lines



Dataset

Source: customers.xlsx
Fields: Date, Region, Product, Quantity, UnitPrice, StoreLocation, CustomerType, Discount, Salesperson, TotalPrice, PaymentMethod, Promotion, Returned, OrderID, CustomerName, ShippingCost, OrderDate, DeliveryDate, RegionManager.


Deployment

Hosted on Streamlit Cloud.
Auto-sync Excel → SQL on first login.
Role-based access control with SHA-256 password hashing.


Future Enhancements

Migrate to PostgreSQL for persistent storage.
Advanced forecasting (ARIMA/Prophet).

Automated KPI reporting via email.
Sentiment analysis on promotions/feedback.


GitHub Repository
https://github.com/vighnesh-shetty-vs/MoneyWiz-CRM-NewUpdate


Sample Users from dataset -

Salesperson -

username - Eva
password - password123

Region Manager -

Salesperson -

username - Eric
password - password123
