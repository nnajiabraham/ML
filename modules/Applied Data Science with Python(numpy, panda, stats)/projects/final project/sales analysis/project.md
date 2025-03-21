In this project, we will analyze the sales data of AAL, a clothing brand in Australia. The data is provided in the file `AusApparalSales4thQrt2020.csv` (relative file path to solution.py file is `modules/Applied Data Science with Python(numpy, panda, stats)/projects/final project/sales analysis/AusApparalSales4thQrt2020.csv`). 

Some rules for implementing the project:
- You can only use the following libraries:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scipy
    - plotly
- Use Plan.md to write out a step by step plan for implementing the project.
- Don't make any assumptions of the data. Use variables to dynamically get information from the data as you create the analysis. If you can't use a variable, write out a comment to explain the assumption you are making for when implementing the code for any analysis.
- Each Data wrangling, Data analysis, Data visualization, Report generation step should be in a separate function. We want to be able to run each step independently and verify the output.
- For each function, write out the steps you will take to complete the function.
- For each step, write out the code you will use to complete the step.
- For each code block, write out a comment to explain the code.
- For each plot, write out the code you will use to create the plot.
- For each table, write out the code you will use to create the table.
- For each analysis, write out the code you will use to complete the analysis.
- For each recommendation, write out the code you will use to complete the recommendation.
- In order to ensure solution.py doesn't get too large, you can create other files to help with the implementation but make sure solution.py is the only file that is run and the functions in the other files are categorized by the step they are used in or the action they are performing. For example, you can create a file called `data_wrangling.py` to help with the data wrangling step.
- We want to keep track of each step completed and the results, so after implementing each substep or step in the plan.md, mark them as completed in the plan.md file then move to the next step.
- After each step, write out a {step_number}_result.md file to in the checkpoints folder to record the results of the step, completion status, actions taken, actions to be taken next, and any other relevant information that will help you with resuming if failures or issues arise in the implementation of the current step or help guide you for whats needed in next step.
- Create a final_report.md file to document the final report and recommendations based on the analysis.


Here is the project statement below:
```
Sales Analysis

Project statement:

AAL, established in 2000, is a well-known brand in Australia, particularly recognized for its clothing business. It has opened branches in various states, metropolises, and tier-1 and tier-2 cities across the country.

The brand caters to all age groups, from kids to the elderly.

Currently experiencing a surge in business, AAL is actively pursuing expansion opportunities. To facilitate informed investment decisions, the CEO has assigned the responsibility to the head of AAL’s sales and marketing (S&M) department. The specific tasks include:

Identify the states that are generating the highest revenues.
Develop sales programs for states with lower revenues. The head of sales and marketing has requested your assistance with this task.
Analyze the sales data of the company for the fourth quarter in Australia, examining it on a state-by-state basis. Provide insights to assist the company in making data-driven decisions for the upcoming year.

*Enclosed is the CSV (AusApparalSales4thQrt2020.csv) file that covers the said data.

Perform the following steps:

As a data scientist, you must perform the following steps on the enclosed data:

Data wrangling
Data analysis
Data visualization
Report generation
Data wrangling
Ensure that the data is clean and free from any missing or incorrect entries.
Inspect the data manually to identify missing or incorrect information using the functions isna() and notna().
Based on your knowledge of data analytics, include your recommendations for treating missing and incorrect data (dropping the null values or filling them).
Choose a suitable data wrangling technique—either data standardization or normalization. Execute the preferred normalization method and present the resulting data. (Normalization is the preferred approach for this problem.)
Share your insights regarding the application of the GroupBy() function for either data chunking or merging, and offer a recommendation based on your analysis.
Data analysis
Perform descriptive statistical analysis on the data in the Sales and Unit columns. Utilize techniques such as mean, median, mode, and standard deviation for this analysis.
Identify the group with the highest sales and the group with the lowest sales based on the data provided.
Identify the group with the highest and lowest sales based on the data provided.
Generate weekly, monthly, and quarterly reports to document and present the results of the analysis conducted.
(Use suitable libraries such as NumPy, Pandas, and SciPy for performing the analysis.)

Data visualization
Use suitable data visualization libraries to construct a dashboard for the head of sales and marketing. The dashboard should encompass key parameters:
State-wise sales analysis for different demographic groups (kids, women, men, and seniors).
Group-wise sales analysis (Kids, Women, Men, and Seniors) across various states.
Time-of-the-day analysis: Identify peak and off-peak sales periods to facilitate strategic planning for S&M teams. This information aids in designing programs like hyper-personalization and Next Best Offers to enhance sales.
Ensure the visualization is clear and accessible for effective decision-making by the head of sales and marketing (S&M).
The dashboard must contain daily, weekly, monthly, and quarterly charts.

Only seaborn is preferred for the visualization since it is a statistical visualization library.

Include your recommendation and indicate why you are choosing the recommended visualization package.
Report generation
Use suitable graphs, plots, and analysis reports in the report, along with recommendations. Note that various aspects of analysis require different graphs and plots.
Use a box plot for descriptive statistics.
Use the Seaborn distribution plot for any other statistical plotting.
```