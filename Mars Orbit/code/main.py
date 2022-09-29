import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
from datetime import datetime

def get_times(data):
    year = data[:, 0]
    month = data[:, 1]
    day = data[:, 2]
    hour = data[:, 3]
    minute = data[:, 4]
    time0 = datetime(year[0], month[0], day[0], hour[0], minute[0], 0)
    times = [ ]
    for i in range(0, len(year)):
        timei = datetime(year[i], month[i], day[i], hour[i], minute[i], 0)
        duration = timei - time0
        times.append(duration.total_seconds() / (3600 * 24))
    return np.array(times)
def get_oppositions(data):
    zodiacIndex = data[:, 5]
    degree = data[:, 6]
    minute = data[:, 7]
    second = data[:, 8]
    return zodiacIndex * 30 + degree + minute / 60 + second / (3600)

# q1
def MarsEquantModel(c, r, e1, e2, z, s, times, oppositions):
    c = (np.cos(np.radians(c)), np.sin(np.radians(c)))
    e = (e1 * np.cos(np.radians(e2 + z)), e1 * np.sin(np.radians(e2 + z)))
    observationOffsetAngle = (times * s + z) % 360
    errors = [ ]
    maxErr = float('-inf')
    for i in range(0, len(observationOffsetAngle)):
        tanValue = np.tan(np.radians(observationOffsetAngle[i]))
        ta = (1 + tanValue ** 2)
        tb = -2 * c[0] + 2 * tanValue * (e[1] - c[1] - e[0] * tanValue)
        tc = (e[1] - c[1] - e[0] * tanValue) ** 2 + c[0] ** 2 - r ** 2
        td = np.sqrt(tb ** 2 - 4 * ta * tc)
        tmode = 1 if 0 <= observationOffsetAngle[i] <= 90 or 270 <= observationOffsetAngle[i] <= 360 else -1
        tx = (-tb - td) / (2 * ta) if ((-tb - td) / (2 * ta) * tmode >= 0) else (-tb + td) / (2 * ta)
        ty = e[1] + (tx - e[0]) * tanValue
        terr = np.degrees(np.arctan2(ty, tx)) - ((180 + oppositions[i]) % 360 - 180)
        errors.append(terr)
        maxErr = max(maxErr, abs(terr))
    errors = np.array(errors)
    return errors, maxErr
# q2
def bestOrbitInnerParams(r, s, times, oppositions):
    c = 10
    e1 = 1
    e2 = 10
    z = 0
    for i in range(0, 3):
        z = min(list(map(lambda x : (x, MarsEquantModel(c, r, e1, e2, x, s, times, oppositions)[1]), np.linspace(0, 360, 360))), key = lambda x : x[1])[0]
        e2 = min(list(map(lambda x : (x, MarsEquantModel(c, r, e1, x, z, s, times, oppositions)[1]), np.linspace(z, 360, 360))), key = lambda x : x[1])[0]
        c = min(list(map(lambda x : (x, MarsEquantModel(x, r, e1, e2, z, s, times, oppositions)[1]), np.linspace(0, 360, 360))), key = lambda x : x[1])[0]
        e1 = min(list(map(lambda x : (x, MarsEquantModel(c, r, x, e2, z, s, times, oppositions)[1]), np.linspace(0, 0.5 * r, 300))), key = lambda x : x[1])[0]
    result = sp.minimize(lambda x : MarsEquantModel(x[0], r, x[1], x[2], x[3], s, times, oppositions)[1], [ c, e1, e2, z ], method='Nelder-Mead', options={'xatol' : 1e-5 ,'disp':False, 'return_all' :False})
    c, e1, e2, z = result.x
    errors, maxErr = MarsEquantModel(c, r, e1, e2, z, s, times, oppositions)
    return c, e1, e2, z, errors, maxErr
# q3
def bestS(r, times, oppositions):
    split = 10
    s = 0
    sRange = np.array([360 / 686, 360 / 688])
    for i in range(0, 7):
        sRange = min(list(map(lambda x : (x, x - ((sRange[1] - sRange[0]) / split), x + ((sRange[1] - sRange[0]) / split), bestOrbitInnerParams(r, x, times, oppositions)[5]), np.linspace(sRange[0], sRange[1], split))), key = lambda x : x[3])[0 : 3]
        s = sRange[0]
        sRange = sRange[1 : 3]
    c, e1, e2, z, errors, maxErr = bestOrbitInnerParams(r, s , times, oppositions)
    return s, errors, maxErr
# q4
def bestR(s, times, oppositions):
    initR = 7
    rs = [ ]
    incBy = 0.15
    r = initR
    for outerLoopC in range(6):
        for innerLoopCounter in range(10):
            if r <= 7 or r >= 9:
                break
            c, e1, e2, z, errors, maxErr = bestOrbitInnerParams(r, s, times, oppositions)
            rs.append((r, maxErr))
            c = (np.cos(np.radians(c)), np.sin(np.radians(c)))
            e = (e1 * np.cos(np.radians(e2 + z)), e1 * np.sin(np.radians(e2 + z)))
            observationOffsetAngle = (times * s + z) % 360
            xDistances = (e[1] - e[0] * np.tan(np.radians(observationOffsetAngle))) / (np.tan(np.radians(oppositions)) - np.tan(np.radians(observationOffsetAngle)))
            yDistances = xDistances * np.tan(np.radians(oppositions))
            distances = np.sqrt((yDistances - c[0]) ** 2 + (yDistances - c[1]) ** 2)
            r = np.mean(distances)
        incBy = incBy + incBy
        r = initR + incBy
    r = min(rs, key = lambda x : x[1])[0]
    c, e1, e2, z, errors, maxErr = bestOrbitInnerParams(r, s, times, oppositions)
    return r, errors, maxErr
# q5
def bestMarsOrbitParams(times, oppositions):
    r = 6
    s = 360 / 687
    maxErr = 1
    itteration = 0
    while maxErr > (4 / 60):
        r, errors, maxErr = bestR(s, times, oppositions)
        s, errors, maxErr2 = bestS(r, times, oppositions)
        itteration = itteration + 1
    c, e1, e2, z, errors, maxErr = bestOrbitInnerParams(r, s, times, oppositions)
    return r, s, c, e1, e2, z, errors, maxErr


if __name__ == "__main__":

    # Import oppositions data from the CSV file provided
    data = np.genfromtxt(
        "../data/01_data_mars_opposition_updated.csv",
        delimiter=",",
        skip_header=True,
        dtype="int",
    )
    # Extract times from the data in terms of number of days.
    # "times" is a numpy array of length 12. The first time is the reference
    # time and is taken to be "zero". That is times[0] = 0.0
    times = get_times(data)
    assert len(times) == 12, "times array is not of length 12"

    # Extract angles from the data in degrees. "oppositions" is
    # a numpy array of length 12.
    oppositions = get_oppositions(data)
    assert len(oppositions) == 12, "oppositions array is not of length 12"
    
    
    # Call the top level function for optimization
    # The angles are all in degrees
    r, s, c, e1, e2, z, errors, maxError = bestMarsOrbitParams(
        times, oppositions
    )

    assert max(list(map(abs, errors))) == maxError, "maxError is not computed properly!"
    print(
        "Fit parameters: r = {:.4f}, s = {:.4f}, c = {:.4f}, e1 = {:.4f}, e2 = {:.4f}, z = {:.4f}".format(
            r, s, c, e1, e2, z
        )
    )
    print("The maximum angular error = {:2.4f}".format(maxError))