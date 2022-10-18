# -*- coding: utf-8 -*-

# 生肖合婚
# http://www.ibazi.cn/article/39858
# https://www.ddnx.com/shenghuo/242712.html


from __future__ import absolute_import, unicode_literals


zodiac_match = {
    '鼠': {
        '宜': (['龙', '猴', '牛'], '大吉，心心相印，富贵幸福，万事易成功，终身。其他属相次之。'),
        '忌': (['马', '兔', '羊'], '不能富家，灾害并至，凶煞重重，甚至骨肉分离，不得安宁。'),
        '解释': '子鼠与丑牛六合，因此最宜找个属牛的对象，此乃上上等婚配。其次是与申猴辰龙三合，故也宜找个属猴或属龙的，此乃上等婚配。子鼠与午马相冲，因此最忌找个属马的，此乃下下等婚配。子鼠又与卯兔相刑，故也忌找属兔的，此乃下等婚配。子鼠与未羊也有相害的成分，故也不宜找属羊的，此乃中下等婚配。有时也讲子子自刑，此乃中下等婚配，故也应避免同属相的。'
    }, 
    '牛': {
        '宜': (['鼠', '蛇', '鸡'], '大吉，天做良缘，家道大旺，财盛人兴。'),
        '忌': (['马', '羊', '狗'], '吉凶各有，甘苦共存，无进取心，内心多忧疑苦惨。'),
        '解释': '丑牛与子鼠六合，因此最宜找个属鼠的对象，此乃上上等婚配。其次是与巳蛇酉鸡三合，故也宜找个属蛇或属鸡的，此乃上等婚配。丑牛与未羊相冲，因此最忌找属羊的，此乃下下等婚配。丑牛又与未羊、戌狗构成三刑，故最忌找属羊属狗的，此乃下等婚配。丑牛又与午马相害，故也不宜找属马的，此乃中下等婚配。'
    },
    '虎': {
        '宜': (['马', '狗', '猪'], '大吉，同心永结，德高望重，家业终成，富贵荣华，子孙昌盛。'),
        '忌': (['蛇', '猴'], '夫妻不和，忧愁不断，有破财之兆，空虚寂寞。'),
        '解释': '寅虎与亥猪六合，因此最宜找个属猪的对象，此乃上上等婚配。其次是与午马戌狗三合，此乃上等婚配，故也宜找个属马或属狗的。寅虎与申猴相冲，因此最忌找属猴的，此乃下下等婚配。寅虎又与巳蛇、申猴构成三刑，故最忌找属蛇属猴的，此乃下等婚配。'
    },
    '兔': {
        '宜': (['羊', '狗', '猪'], '功业成就，安居乐业，专利兴家。'),
        '忌': (['龙', '鼠', '鸡'], '家庭难有幸福，逆境之象。'),
        '解释': '卯兔与戌狗六合，因此最宜找个属狗的对象，此乃上上等婚配。其次是与亥猪未羊三合，故也宜找个属猪或属羊的，此乃上等婚配。卯兔与酉鸡相冲，因此最忌找属鸡的，此乃下下等婚配。卯兔与子鼠又有相刑的成分，故也不宜找属鼠的，此乃中下等婚配。卯兔还与辰龙相害，故也不宜找属龙的，此乃中下等婚配。'
    },
    '龙': {
        '宜': (['鼠', '猴', '鸡'], '大吉绨结良缘，勤俭发家，日刡昌盛，富贵成功，子孙继世。'),
        '忌': (['狗'], '不能和睦终世，破坏离别，不得心安。'),
        '解释': '辰龙与酉鸡六合，因此最宜找个属鸡的对象，此乃上上等婚配。其次是与申猴子鼠三合，故也宜找个属猴属鼠的，此乃上等婚配。辰龙与戌狗相冲，因此最忌找属狗的，此乃下下等婚配。辰龙与卯兔又有相害的成分，故也不宜找属兔的，此乃中下等婚配。有时也讲辰辰自刑，故也要注意避免同属相的，此乃中下等婚配。'
    },
    '蛇': {
        '宜': (['猴', '牛', '鸡'], '大吉祥，此中属相相配为福禄鸳鸯，智勇双全，功业垂成，足立宝地，名利双收，一生幸福。'),
        '忌': (['猪', '虎'], '家境虽无大的困苦和失败，但夫妻离心离德，子息缺少，灾厄百端，晚景不祥。'),
        '解释': '巳蛇与申猴六合，因此最宜找个属猴的对象，此乃上上等婚配。其次是与酉鸡丑牛三合，故也宜找个属鸡或属牛的，此乃上等婚配。巳蛇与亥猪相冲，因此最忌找属猪的，此乃下下等婚配。巳蛇与寅虎相刑，此乃下等婚配;有时也讲寅、巳、申三刑，故三口人不宜构成蛇、虎、猴的格局。'
    },
    '马': {
        '宜': (['虎', '羊', '狗'], '大吉，夫妻相敬，紫气东来，福乐安详，家道昌隆。'),
        '忌': (['鼠', '牛'], '中年运气尚可，病弱短寿，难望幸福，重生凶兆，一生辛苦，配偶早丧，子女别离。'),
        '解释': '午马与未羊六合，因此最宜找个属羊的对象，此乃上上等婚配。其次是与寅虎戌狗三合，故也宜找个属虎属猴的，此乃上等婚配。午马与子鼠相冲，因此最忌找属鼠的，此乃下下等婚配。午马与丑牛相害，因此也应避免找属牛的，此乃中下等婚配。有时又讲午午自刑，故也应避免同属相的，此乃中下等婚配。'
    },
    '羊': {
        '宜': (['兔', '马', '猪'], '天赐良缘，家道谐和，大业成而有德望。'),
        '忌': (['牛', '狗', '羊'], '夫妻一生难得幸福，多灾多难，一生劳碌，早失配偶或子孙。'),
        '解释': '未羊与午马六合，因此最宜找个属马的对象，此乃上上等婚配。其次是与亥猪卯兔三合，故也宜找个属猪属兔的，此乃上等婚配。未羊与丑牛相冲，因此最忌找属牛的，此乃下下等婚配。未羊与丑牛戌狗构成三刑，因此也不宜找属狗的，此乃下等婚配。未羊与子鼠相害，因此也不宜找属鼠的，此乃中下等婚配。'
    },
    '猴': {
        '宜': (['鼠', '蛇', '龙'], '此中属相相配为珠联璧合，一帆风顺，富贵成功，子孙兴旺。'),
        '忌': (['虎', '猪'], '灾害多起，晚景尚可，但恐寿不到永，疾病困难。'),
        '解释': '申猴与巳蛇六合，因此最宜找个属蛇的对象，此乃上上等婚配。其次是与子鼠辰龙三合，故也宜找个属鼠属龙的，此乃上等婚配。申猴与寅虎相冲，因此最忌找属虎的，此乃下下等婚配。申猴又与巳蛇寅虎构成三刑，因此三口人之间不宜形成虎、蛇、猴这种配合。申猴又与亥猪相害，因此也不宜找属猪的，此乃中下等婚配。'
    },
    '鸡': {
        '宜': (['牛', '龙', '蛇'], '此中属相相配祥开白事，有天赐之福，并有名望，功利荣达，家事亨通。'),
        '忌': (['兔', '狗'], '金鸡玉犬难逃避，合婚双份不可迁，多灾多难。'),
        '解释': '酉鸡与辰龙六合，因此最宜找个属龙的对象，此乃上上等婚配。其次是与巳蛇丑牛三合，故也宜找个属蛇属牛的，此乃上等婚配。酉鸡与卯兔相冲，因此最忌找属兔的，此乃下下等婚配。酉鸡又与戌狗相害，故也不宜找属狗的，此乃中下等婚配。' 
    },
    '狗': {
        '宜': (['虎', '兔', '马'], '大吉，天做之合，处处成功，福碌永久，家运昌隆。'),
        '忌': (['羊', '龙', '鸡', '牛'], '灾害垒起，钱财散败，一生艰辛，事与愿违。'),
        '解释': '戌狗与卯兔六合，因此最宜找个属兔的对象，此乃上上等婚配。其次是与寅虎午马三合，故也宜找个属虎属马的，此乃上等婚配。戌狗与辰龙相冲，故最忌找属龙的，此乃下下等婚配。戌狗又与未羊丑牛构成三刑，故不宜找属羊属牛的，此乃下等婚配。戌狗又与酉鸡相害，故不宜找属鸡的，此乃中下等婚配。'
    },
    '猪': {
        '宜': (['羊', '兔', '虎'], '大吉，五事其昌，安富尊荣，子孙健壮，积财多福。'),
        '忌': (['猴', '蛇'], '猪猴不到头，朝朝日日泪交流，比能共长久，终生难于幸福。'),
        '解释': '亥猪与寅虎六合，因此最宜找个属虎的对象，此乃上上等婚配。其次是与卯兔未羊三合，故也宜找个属兔属羊的，此乃上等婚配。亥猪与巳蛇相冲，故最忌找属蛇的，此乃下下等婚配。亥猪又与申猴相害，因此也不宜找属猴的，此乃中下等婚配。有时也讲亥亥自刑，故也不宜找同属相的，此乃中下等婚配。',
    }
}
