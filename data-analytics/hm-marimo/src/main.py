import marimo

__generated_with = "0.12.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo, num):
    mo.md(f"The value of num is {num}")
    return


@app.cell
def _():
    num = 42
    return (num,)


@app.cell
def _(mo):
    icon = mo.ui.dropdown(["🍃", "🌊", "✨"], value="🍃")
    return (icon,)


@app.cell
def _(icon, mo):
    repetition = mo.ui.slider(1, 16, label=f"number of {icon.value}: ")
    return (repetition,)


@app.cell
def _(icon, repetition):
    icon, repetition
    return


@app.cell(hide_code=True)
def _(icon, mo, repetition):
    mo.md(icon.value * repetition.value)
    return


if __name__ == "__main__":
    app.run()
