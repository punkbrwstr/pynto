import datetime
import pandas as pd
from typing import Union


datelike: type = Union[int,str,datetime.date,datetime.datetime,pd.Timestamp]

def now() -> datetime.date:
    d = datetime.datetime.utcnow()
    if d.weekday() < 5 and d.hour >= 22:
        return offset('B', d, 1)
    else:
        return d.date()

def parse_date(date: datelike) -> datetime.date:
    if isinstance(date, datetime.date):
        return date       
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
        return date.date()
    elif isinstance(date, str):
        return datetime.datetime.strptime(date, "%Y-%m-%d").date()
    else:
        raise TypeError(f'{type(date)} is not datelike')

def round_b(date: datelike) -> datetime.date:
    date = parse_date(date)
    weekday = date.weekday()
    if weekday < 5:
        return date
    else:
        return date + datetime.timedelta(days=7 - weekday)

def round_w_fri(date: datelike) -> datetime.date:
    date = parse_date(date)
    weekday = date.weekday()
    if weekday == 4:
        return date
    else:
        return date + datetime.timedelta(days=4 - weekday)

def round_w_tue(date: datelike) -> datetime.date:
    date = parse_date(date)
    weekday = date.weekday()
    if weekday == 1:
        return date
    elif weekday == 0:
        return date + datetime.timedelta(days=1)
    else:
        return date + datetime.timedelta(days=8 - weekday)

def round_bm(date: datelike) -> datetime.date:
    date = parse_date(date)
    next_month = date.replace(day=28) + datetime.timedelta(days=4) 
    last_day = next_month - datetime.timedelta(days=next_month.day)
    return last_day - datetime.timedelta(days=max(0,last_day.weekday() - 4))

def round_ba_dec(date: datelike) -> datetime.date:
    date = parse_date(date)
    date = date +  relativedelta(months=12-date.month)
    return round_bm(date)

def round_bq_dec(date: datelike) -> datetime.date:
    date = parse_date(date)
    remainder = date.month % 3
    plusmonths = 0 if remainder == 0 else 2 // remainder
    date = date +  relativedelta(months=plusmonths)
    return round_bm(date)

def add_bm(date: datelike, months: int) -> datetime.date:
    month = date.month - 1 + months
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, calendar.monthrange(year,month)[1])
    day = datetime.date(year, month, day)
    return round_bm(day)

def add_b(date: datetime.date, b: int) -> datetime.date:
    daysSinceMonday = date.weekday()
    prevMonday = date - datetime.timedelta(days=daysSinceMonday)
    weekendsInDistance = (b + daysSinceMonday) // 5
    return prevMonday + datetime.timedelta(days=b + weekendsInDistance * 2 + daysSinceMonday)

def count_b(d1: datetime.date, d2: datetime.date) -> int:
    daysSinceMonday1 = d1.weekday()
    prevMonday1 = d1 - datetime.timedelta(days=daysSinceMonday1)
    daysSinceMonday2 = d2.weekday()
    prevMonday2 = d2 - datetime.timedelta(days=daysSinceMonday2)
    days = (prevMonday2 - prevMonday1).days
    days -= days * 2 // 7
    return days - daysSinceMonday1 + daysSinceMonday2

