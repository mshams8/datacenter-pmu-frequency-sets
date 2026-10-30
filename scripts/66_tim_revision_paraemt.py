"""Retained forcing definitions; physical reruns are outside this replay package."""
import math

class ChirpProfile:
    def __init__(self, f, beta, amplitude):
        self.f, self.beta, self.amplitude = f, beta, amplitude
    def value(self, t):
        u = max(t - .5, 0.)
        envelope = .5 - .5 * math.cos(math.pi * min(u/.1, 1.))
        # f is the instantaneous frequency at the 1.75-s analysis-window centre.
        phase = 2*math.pi*((self.f - 1.25*self.beta)*u + .5*self.beta*u*u)
        return 1 + self.amplitude*envelope*math.sin(phase)

def cases():
    rows=[]
    for bus, condition in [(4,'op_0p95_intact'), (39,'op_1p05_line_6_11_out')]:
        for f in [14.7,29.,31.,45.3]:
            for offset,beta in [(0,0),(-.25,0),(.25,0),(0,-.25),(0,.25)]:
                rows.append(dict(case_id=f'b{bus}_f{f:g}_o{offset:g}_b{beta:g}',bus=bus,
                                 condition=condition,nominal=f,offset=offset,beta=beta,amplitude=.02))
    return rows
