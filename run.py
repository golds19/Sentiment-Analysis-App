from app import create_app

if __name__ == "__main__":
    print("running sentiment app")
    app = create_app()
    app.run(debug=True)