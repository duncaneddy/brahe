import brahe

def test_epoch_range():
    epcs = brahe.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, brahe.TAI)
    epcf = brahe.Epoch.from_datetime(2022, 1, 2, 0, 0, 0.0, 0.0, brahe.TAI)
    epc  = brahe.Epoch.from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, brahe.TAI)

    epcv = []
    for e in brahe.TimeRange(epcs, epcf, 1.0):
        assert epc == e
        epc += 1
        epcv.append(e)


    epcl = brahe.Epoch.from_datetime(2022, 1, 1, 23, 59, 59.0, 0.0, brahe.TAI)
    assert len(epcv) == 86400
    assert (epcv[-1] != epcf)
    assert (epcv[-1] - epcl) <= 1.0e-9