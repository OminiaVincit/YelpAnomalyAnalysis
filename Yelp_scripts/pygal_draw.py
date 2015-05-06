import pygal

pie_chart = pygal.Pie()
pie_chart.title = 'Portion of subtopics in review'
pie_chart.add('Japanese', 34.29)
pie_chart.add('Location', 31.29)
pie_chart.add('Desert', 10.6)
pie_chart.add('Noodle', 10.5)
pie_chart.add('Seafood', 10.2)
pie_chart.add('Other', 2.12)
pie_chart.render_to_file('review_topic.svg')