# Standalone plots from saved fitted estimates

This folder plots saved state estimates for captures 5 and 6. It does not include raw detection pickle files or model coefficients and does not fit a model or calculate new state estimates.

Run with Python 3 and NumPy, SciPy, and Matplotlib installed:

```bash
python plot_captures.py
```

The script reads only `results/fitted_estimates.pkl` and opens both figures in interactive Matplotlib windows. It does not save figure image files. The pickle contains fitted gaze and accommodation, capture 1 linear gaze, error in arcminutes, and elapsed time for every row of both captures. Plot outliers are masked with a centered 0.3-second local median; saved fitted estimate arrays remain unchanged. Row masks are written as pickle files.

Plot formatting is fixed to 20 pt titles, 18 pt axis and tick labels, 20 × 9 inch figures, gaze limits −12.5 to 7.5 degrees, accommodation limits 0 to 4.5 D, and error limits −50 to 50 arcminutes. Lines have gaps at masked spike samples.

Install plotting dependencies using `python -m pip install -r requirements.txt`. The interactive window uses the Tk backend; run in a graphical desktop session.
