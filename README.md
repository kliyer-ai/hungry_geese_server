# hungry_geese_server

# Setup
The setup below works works UNIX like system. Windows should work in a similar fashion. Just give it a quick google.
```bash
python3 -m venv <directory name>
source <directory name>/bin/activate
pip install -r requirements.txt
export FLASK_APP=app.py
export FLASK_ENV=development
```

# Start the server:
```
python3 run.py
```
Then navigate to localhost:5000 to see the game. You can start playing by hitting one of the arrow keys.

# Enemy player
You can change the enemy players [in line 14](https://github.com/kliyer-ai/hungry_geese_server/blob/d708703c83f91f07e790719717d7cdb7408d8489/app/__init__.py).
