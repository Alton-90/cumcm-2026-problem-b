"""Student-t mean intervals with standard-library arithmetic only."""
import math,statistics

def _betacf(a,b,x):
    tiny=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.;d=1-qab*x/qap
    if abs(d)<tiny:d=tiny
    d=1/d;h=d
    for m in range(1,401):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1+aa*d
        if abs(d)<tiny:d=tiny
        c=1+aa/c
        if abs(c)<tiny:c=tiny
        d=1/d;h*=d*c;aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1+aa*d
        if abs(d)<tiny:d=tiny
        c=1+aa/c
        if abs(c)<tiny:c=tiny
        d=1/d;delta=d*c;h*=delta
        if abs(delta-1)<3e-14:return h
    raise ArithmeticError('Incomplete beta fraction did not converge')

def _beta_regularized(x,a,b):
    if x<=0:return 0.
    if x>=1:return 1.
    factor=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x<(a+1)/(a+b+2):return factor*_betacf(a,b,x)/a
    return 1-factor*_betacf(b,a,1-x)/b

def t975(df):
    if not isinstance(df,int) or df<1:raise ValueError('positive integer degrees of freedom required')
    def cdf(t):return 1-.5*_beta_regularized(df/(df+t*t),df/2,.5)
    lo,hi=0.,1.
    while cdf(hi)<.975:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if cdf(mid)<.975:lo=mid
        else:hi=mid
    return (lo+hi)/2

def summarize(rows):
    vals=[float(r['value']) for r in rows];n=len(vals)
    mean=statistics.mean(vals);sd=statistics.stdev(vals);critical=t975(n-1);half=critical*sd/math.sqrt(n)
    sorted_values=sorted(vals);pos=(n-1)*.95;i=int(pos);p95=sorted_values[i]+(pos-i)*(sorted_values[min(i+1,n-1)]-sorted_values[i])
    return dict(n=n,mean=mean,sd=sd,ci95=[mean-half,mean+half],ci_method='Student t, case-level mean',t_critical=critical,median=statistics.median(vals),p95=p95,min=min(vals),max=max(vals),sources=sum(int(r['N']) for r in rows),cleared=sum(int(r['cleared']) for r in rows),pooled=sum(float(r['total']) for r in rows)/sum(int(r['N']) for r in rows))
