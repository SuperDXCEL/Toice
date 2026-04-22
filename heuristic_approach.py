import sys
import time
import string

ONE_POINT = [
    "ahora mismo", "inmediatamente", "urgente", "no hay tiempo", "hoy mismo",
    "última oportunidad", "antes de que cierre",
    "Hacienda", "Seguridad Social", "Policía Nacional", "Guardia Civil",
    "Banco de España", "SEPE", "Correos", "Endesa", "Iberdrola", "Movistar",
    "multa", "deuda pendiente", "expediente abierto", "proceso judicial",
    "inhabilitación", "cuenta bloqueada",
    "transferencia", "criptomonedas", "Bitcoin", "ingreso inmediato",
    "reembolso pendiente", "tarjeta regalo",
    "ha sido seleccionado", "ha ganado", "le corresponde", "paquete retenido",
    "herencia", "sorteo", "premio pendiente",
    "un familiar", "ha tenido un accidente", "está en el hospital",
]

TWO_POINTS = [
    "no lo comente con nadie", "no cuelgue", "quédese en línea",
    "no llame a su banco", "es confidencial", "no se lo diga a su familia",
    "entre nosotros",
    "orden de arresto", "embargo", "está detenido", "necesita fianza",
    "número de tarjeta", "CVV", "clave secreta", "PIN",
    "Agencia Tributaria", "Bizum",
    "su nieto",
]

def heuristic_system(full_text, curr_text, score, start_timestamp_of_convo, websocket):
    """
        Main function for the first layer of protection, a simple dictionary
        approach.
        If score > 3 && score < 5, move on to the second layer of protection, .
        If score > 5, sound alarm.
    """
    print("CURRENT score: ", score)
    print("SECONDS ELAPSED: ", time.time() - start_timestamp_of_convo)
    if score > 2 and score < 4 and time.time() - start_timestamp_of_convo > 60:
        score = second_layer(full_text, score)
    elif score > 4:
        score = sound_alarm(score, websocket)
    else:
        score = first_layer(curr_text, score)
    return score

def first_layer(text, score=0):
    """
        For keyword in ONE_POINT, score +1
        For keyword in TWO_POINTS, score +2
    """
    text = clean_text(text)
    for phrase in TWO_POINTS:
        if phrase.lower() in text.lower():
            score += 2
    for phrase in ONE_POINT:
        if phrase.lower() in text.lower():
            score += 1
    return score

def clean_text(text):
    """
        Remove punctuation
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def sound_alarm(score, websocket):
    print("ALARM!!!", score)
    websocket.send("ALARM")

def second_layer(fullText, score):
    return score + 1
