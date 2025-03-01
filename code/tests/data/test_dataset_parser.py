import qut01.data.split_utils


def test_merge_clusters():
    clusters = [[8], [1, 2, 3], [3, 4, 5], [7, 8], [6, 7, 8], [10], [1]]
    merged_clusters = qut01.data.split_utils._merge_clusters(clusters)
    merged_clusters.sort()
    for cluster in merged_clusters:
        cluster.sort()
    assert merged_clusters == [[1, 2, 3, 4, 5], [6, 7, 8], [10]]
