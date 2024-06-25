from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as pyo
import numpy as np
from scipy.stats import gaussian_kde

app = Flask(__name__)

# List of probability distributions
distributions = ["Bernoulli", "Binomial", "Poisson", "Normal", 
                 "Standard Normal", "Student's t", "Uniform", 
                 "Log Normal", "Chi-Square", "F-Distribution"]


@app.route('/')
def home():
    return render_template('home.html', distributions=distributions)


@app.route('/distribution/<dist_name>', methods=['GET', 'POST'])
def distribution(dist_name):
    plot_div = ""
    params = None

    if request.method == 'POST':
        params = request.form
        plot_div = generate_plot(dist_name, params)
    return render_template('distribution.html', dist_name=dist_name, plot_div=plot_div, params=params)


#function to generate plots
def generate_plot(dist_name, params):
    size = int(params.get('size', 1000))
    
    if dist_name == "Bernoulli":
        n = 1
        p = float(params.get('p', 0.5))
        data = np.random.binomial(n, p, size)
    
    elif dist_name == "Binomial":
        n = int(params.get('n', 10))
        p = float(params.get('p', 0.5))
        data = np.random.binomial(n, p, size)
    
    elif dist_name == "Poisson":
        lam = float(params.get('lam', 1))
        data = np.random.poisson(lam, size)
    
    elif dist_name == "Normal":
        loc = float(params.get('loc', 0))
        scale = float(params.get('scale', 1))
        data = np.random.normal(loc, scale, size)
    
    elif dist_name == "Standard Normal":
        data = np.random.normal(loc=0, scale=1, size=size)
    
    elif dist_name == "Student's t":
        df = float(params.get('df', 10))
        data = np.random.standard_t(df, size)
    
    elif dist_name == "Uniform":
        low = float(params.get('low', 0))
        high = float(params.get('high', 1))
        data = np.random.uniform(low, high, size)
    
    elif dist_name == "Log Normal":
        mean = float(params.get('mean', 0))
        sigma = float(params.get('sigma', 1))
        data = np.random.lognormal(mean, sigma, size)
    
    elif dist_name == "Chi-Square":
        df = float(params.get('df', 2))
        data = np.random.chisquare(df, size)
    
    elif dist_name == "F-Distribution":
        dfnum = float(params.get('dfnum', 2))
        dfden = float(params.get('dfden', 2))
        data = np.random.f(dfnum, dfden, size)
    
    #create charts for different distributions
    if dist_name == "Bernoulli" or dist_name == "Binomial" or dist_name == "Poisson":
        fig = go.Figure(data=[go.Histogram(x=data, nbinsx=20)])
        fig.update_layout(
            title=f'{dist_name} Distribution', xaxis_title='Value', yaxis_title='Probability',
            template='ggplot2', width=800, height=600)
        plot_div = pyo.plot(fig, output_type='div')
    else:
        fig = ff.create_distplot([data], [dist_name], curve_type='kde', colors=['blue'])
        fig.update_layout(
            title=f'{dist_name}', xaxis_title='Value', yaxis_title='Density',
            template='ggplot2', width=800, height=600)
        plot_div = pyo.plot(fig, output_type='div')
        
    return plot_div

if __name__ == '__main__':
    app.run(debug=True)
