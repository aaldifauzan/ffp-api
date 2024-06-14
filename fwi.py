# This code taken from:
# "Updated source code for calculating fire danger indices in the
# canadian forest fire weather index system. 2015" : Y. Wang, K.R. Anderson,
# and R.M. Suddaby.
# As provided by Natural Resources Canada (NRCan)
# This reproduction has not been produced in affiliation with,
# or with the endorsement of, NRCan.

import math


# Define Class FWI Class first
class FWICLASS:

    def __init__(self, temp, rhum, wind, prcp):
        self.h = rhum
        self.t = temp
        self.w = wind
        self.p = prcp

    def FFMCcalc(self, ffmc0):
        # *Eq. 1* #
        mo = (147.2*(101.0 - ffmc0))/(59.5 + ffmc0)
        if (self.p > 0.5):
            # *Eq. 2* #
            rf = self.p - 0.5
            if (mo > 150.0):
                # *Eq. 3b*#
                mo = (mo+42.5*rf*math.exp(-100.0/(251.0-mo)) * (1.0 - math.exp(-6.93/rf))) \
                    + (.0015*(mo - 150.0)**2) * \
                    math.sqrt(
                        rf)
            elif mo <= 150.0:
                # *Eq. 3a*#
                mo = mo+42.5*rf*math.exp(-100.0/(251.0-mo)) * \
                    (1.0 - math.exp(-6.93/rf))
            if(mo > 250.0):
                mo = 250.0
        # *Eq. 4*#
        ed = .942*(self.h**.679) + (11.0*math.exp((self.h-100.0)/10.0))+0.18*(21.1-self.t) \
            * (1.0 - 1.0/math.exp(.1150 * self.h))
        if(mo < ed):
            # *Eq. 5*#
            ew = .618*(self.h**.753) + (10.0*math.exp((self.h-100.0)/10.0)) \
                + .18*(21.1-self.t)*(1.0 - 1.0/math.exp(.115 * self.h))
            if(mo <= ew):
                # *Eq. 7a*#
                kl = .424*(1.0-((100.0 - self.h)/100.0)**1.7)+(.0694*math.sqrt(self.w)) \
                    * (1.0 - ((100.0 - self.h)/100.0)**8)
                # *Eq. 7b*#
                kw = kl * (.581 * math.exp(.0365 * self.t))
                # *Eq. 9*#
                m = ew - (ew - mo)/10.0**kw
            elif mo > ew:
                m = mo
        elif(mo == ed):
            m = mo
        elif mo > ed:
            # *Eq. 6a*#
            kl = .424*(1.0-(self.h/100.0)**1.7)+(.0694*math.sqrt(self.w)) * \
                (1.0-(self.h/100.0) **
                 8)
            # *Eq. 6b*#
            kw = kl * (.581*math.exp(.0365*self.t))
            # *Eq. 8*#
            m = ed + (mo-ed)/10.0 ** kw
        # * Eq. 10*#
        ffmc = (59.5 * (250.0 - m)) / (147.2 + m)
        if (ffmc > 101.0):
            ffmc = 101.0
        if (ffmc <= 0.0):
            ffmc = 0.0
        return ffmc

    def DMCcalc(self, dmc0, mth):
        el = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        t = self.t
        if (t < -1.1):
            t = -1.1
        # *Eqs. 16 and 17*#
        rk = 1.894*(t+1.1) * (100.0-self.h) * (el[mth-1]*0.0001)
        if self.p > 1.5:
            ra = self.p
            # *Eq. 11*#
            rw = 0.92*ra - 1.27
            # *Eq. 12*#
            wmi = 20.0 + 280.0/math.exp(0.023*dmc0)
            if dmc0 <= 33.0:
                # *Eq. 13a*#
                b = 100.0 / (0.5 + 0.3*dmc0)
            elif dmc0 > 33.0:
                if dmc0 <= 65.0:
                    # *Eq. 13b*#
                    b = 14.0 - 1.3*math.log(dmc0)
                elif dmc0 > 65.0:
                    # *Eq. 13c*#
                    b = 6.2 * math.log(dmc0) - 17.2
            # *Eq. 14*#
            wmr = wmi + (1000*rw) / (48.77+b*rw)
            # *Eq. 15*#
            pr = 43.43 * (5.6348 - math.log(wmr-20.0))
        elif self.p <= 1.5:
            pr = dmc0
        if (pr < 0.0):
            pr = 0.0
        dmc = pr + rk
        if (dmc <= 1.0):
            dmc = 1.0
        return dmc

    def DCcalc(self, dc0, mth):
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0,
              2.4, 0.4, -1.6, -1.6]            # *Eq. 22*#
        t = self.t
        if(t < -2.8):
            t = -2.8
        pe = (0.36*(t+2.8) + fl[mth-1])/2
        if pe <= 0.0:
            pe = 0.0
        if (self.p > 2.8):
            ra = self.p
            # *Eq. 18*#
            rw = 0.83*ra - 1.27
            # *Eq. 19*#
            smi = 800.0 * math.exp(-dc0/400.0)
            # *Eqs. 20 and 21*#
            dr = dc0 - 400.0*math.log(1.0+((3.937*rw)/smi))
            if (dr > 0.0):
                dc = dr + pe
            else:
                dc = dc0 + pe
        elif self.p <= 2.8:
            dc = dc0 + pe
        return dc

    def ISIcalc(self, ffmc):
        # *Eq. 1*#
        mo = 147.2*(101.0-ffmc) / (59.5+ffmc)
        # *Eq. 25*#
        ff = 19.115*math.exp(mo*-0.1386) * (1.0+(mo**5.31) /
                                            49300000.0)
        # *Eq. 26*#
        isi = ff * math.exp(0.05039*self.w)
        return isi

    def BUIcalc(self, dmc, dc):
        if dmc <= 0.4*dc:
            # *Eq. 27a*#
            bui = (0.8*dc*dmc) / (dmc+0.4*dc)
        else:
            # *Eq. 27b*
            bui = dmc-(1.0-0.8*dc/(dmc+0.4*dc)) * \
                (0.92+(0.0114*dmc)**1.7)
        if bui < 0.0:
            bui = 0.0
        return bui

    def FWIcalc(self, isi, bui):
        if bui <= 80.0:
            # *Eq. 28a*#
            bb = 0.1 * isi * (0.626*bui**0.809 + 2.0)
        else:
            # *Eq. 28b*#
            bb = 0.1*isi*(1000.0/(25. + 108.64/math.exp(0.023*bui)))
        if(bb <= 1.0):
            # *Eq. 30b*#
            fwi = bb
        else:
            # *Eq. 30a*#
            fwi = math.exp(2.72 * (0.434*math.log(bb)) **
                           0.647)
        return fwi

    def DSRCalc(self, fwi):
        # This formula NOT taken from the 2015 document.
        # This formula taken from "Equations and FORTRAN program for the Canadian Forest Fire Weather Index
        # System. 1985"
        return 0.0272 * math.pow(fwi, 1.77)


# End of class FWI Class
