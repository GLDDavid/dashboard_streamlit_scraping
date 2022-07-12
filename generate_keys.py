import pickle
from pathlib  import Path

import streamlit_authenticator as stauth

names = ["David Gillard", "Mikael Roor"]
usernames = ["dgillard", "mroor"]
passwords = ["XXX", "XXX"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)

# on exécute la commande et on valide une seule fois dans le terminal.
# un fichier hashed_pw doit alors être créé
# il suffit maintenant de remplacer le passwords par des XXXX

# pour info les passwords sont abc123 et def456 mais il faudra supprimer cette ligne