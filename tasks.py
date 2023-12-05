from invoke import task

@task
def run(c):
    """Run the main script."""
    c.run("python3 main.py")


@task
def generate(c):
    """Run the generate script."""
    c.run("python3 test_generation.py")