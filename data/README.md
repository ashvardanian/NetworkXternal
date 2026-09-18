# Datasets

This application accepts data in a shape of adjacency list in CSV format.
The first 2 columns must contain the IDs of source and target nodes.
The third column is optional and can be used for weights.
Here are some relatively big graphs you can use for benchmarks:

- Orkut social network.
  2Gb, `|V|`=3M, `|E|`=117M.
  [Source](http://networkrepository.com/orkut.php).
- Sina Weibo social network.
  2Gb, `|V|`=21M, `|E|`=261M.
  [Source](http://networkrepository.com/soc-sinaweibo.php).
- Twitter social network.
  6Gb, `|V|`=58M, `|E|`=265M.
  [Source](http://networkrepository.com/soc-twitter.php).
- Friendster social network.
  9Gb, `|V|`=65M, `|E|`=1.8B.
  [Source](http://networkrepository.com/soc-friendster.php).
- ClueWeb 2009 network.
  40Gb, `|V|`=1.7B, `|E|`=7.8B.
  [Source](http://networkrepository.com/web-ClueWeb09.php).