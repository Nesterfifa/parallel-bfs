#include <iostream>
#include <vector>
#include <queue>
#include <functional>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <cmath>

using namespace std;

template<typename T>
std::ostream &operator<<(std::ostream &output, const std::vector<T> &data) {
    for (const T &x: data)
        output << x << " ";
    return output;
}

void scan_up(int v, int l, int r, vector<int> &data, vector<int> &tree, int size) {
    if (r - l < 1000000) {
        int sum = 0;
        for (int i = l; i < r; i++) {
            sum += data[i];
        }
        tree[v] = sum;
    } else {
        int m = (l + r) / 2;
#pragma omp task shared(tree)
        {
            scan_up(v * 2 + 1, l, m, data, tree, size);
        }
#pragma omp task shared(tree)
        {
            scan_up(v * 2 + 2, m, r, data, tree, size);
        }
#pragma omp taskwait
        tree[v] = tree[v * 2 + 1] + tree[v * 2 + 2];
    }
}

void scan_down(int v, int l, int r, int left, vector<int> &data, vector<int> &res, vector<int> &tree, int size) {
    if (r - l < 1000000) {
        int sum = left;
        for (int i = l; i < r; i++) {
            sum += data[i];
            res[i] = sum;
        }
    } else {
        int m = (l + r) / 2;
#pragma omp task shared(res)
        {
            scan_down(v * 2 + 1, l, m, left, data, res, tree, size);
        }
#pragma omp task shared(res)
        {
            scan_down(v * 2 + 2, m, r, left + tree[v * 2 + 1], data, res, tree, size);
        }
    }
}

void scan(vector<int> &data, vector<int> &res, vector<int> &tree, int size) {
    {
#pragma omp single
        {
            scan_up(0, 0, size, data, tree, size);
            scan_down(0, 0, size, 0, data, res, tree, size);
        }
    }
}

int filter(vector<int> &data, vector<int> &res, const function<bool(int)> &p, vector<int> &mask, vector<int> &tree,
           int size) {
#pragma omp for
    for (int i = 0; i < size; i++) {
        mask[i] = p(data[i]) ? 1 : 0;
    }
    scan(mask, mask, tree, size);
#pragma omp for
    for (int i = 0; i < size; i++) {
        if (p(data[i])) {
            res[mask[i] - 1] = data[i];
        }
    }
    return mask[size - 1];
}

int bfs_seq(vector<vector<int> > &g, vector<int> &d) {
    d[0] = 0;
    queue<int> q;
    q.push(0);
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        for (auto u: g[v]) {
            if (d[u] == -1) {
                q.push(u);
                d[u] = d[v] + 1;
            }
        }
    }

    return d.back();
}

int parallel_bfs(
        vector<vector<int> > &g,
        vector<int> &frontier,
        vector<int> &new_frontier,
        vector<int> &deg,
        vector<int> &mask,
        vector<int> &tree,
        vector<atomic<int> > &dist) {
    frontier[0] = 0;
    int frontier_size = 1;

#pragma omp parallel shared(deg, frontier, new_frontier, mask, tree, dist)
    {
        while (true) {
#pragma omp for
            for (int i = 0; i < frontier_size; i++) {
                deg[i] = g[frontier[i]].size();
            }
            scan(deg, deg, tree, frontier_size);
            if (deg[frontier_size - 1] == 0) {
                break;
            }
#pragma omp for
            for (int i = 0; i < deg[frontier_size - 1]; i++) {
                new_frontier[i] = -1;
            }
#pragma omp for
            for (int i = 0; i < frontier_size; i++) {
                int v = frontier[i];
                for (int j = 0; j < g[v].size(); j++) {
                    int u = g[v][j];
                    int expect = -1;
                    if (dist[u].compare_exchange_strong(expect, dist[v] + 1)) {
                        new_frontier[i == 0 ? j : deg[i - 1] + j] = u;
                    }
                }
            }
            frontier_size = filter(new_frontier, frontier, [](int x) { return x != -1; }, mask, tree,
                                   deg[frontier_size - 1]);
        }
    }
    return dist.back();
}

vector<vector<int> > generate_graph(int side) {
    vector<vector<int> > g(side * side * side);
    for (int i = 0; i < side; i++) {
        for (int j = 0; j < side; j++) {
            for (int k = 0; k < side; k++) {
                int v = i * side * side + j * side + k;
                if (i + 1 < side) {
                    int u = (i + 1) * side * side + j * side + k;
                    g[v].push_back(u);
                }
                if (j + 1 < side) {
                    int u = i * side * side + (j + 1) * side + k;
                    g[v].push_back(u);
                }
                if (k + 1 < side) {
                    int u = i * side * side + j * side + k + 1;
                    g[v].push_back(u);
                }
            }
        }
    }
    return g;
}

bool equals(vector<int> &a, vector<atomic<int> > &b) {
    for (int i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            cout << i << " " << a[i] << " " << b[i] << endl;
            return false;
        }
    }
    return true;
}

const int side = 400;

void bench() {
    omp_set_num_threads(4);
    omp_set_dynamic(0);
    vector<vector<int> > g = generate_graph(side);
    cout << "constructed" << endl;
    double par_avg = 0, seq_avg = 0;
    for (int t = 0; t < 5; t++) {
        vector<int> dist_seq(g.size(), -1);
        auto start = chrono::steady_clock::now();
        int ans_seq = bfs_seq(g, dist_seq);
        auto stop = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
        cout << "SEQ: " << duration << " ms" << endl;
        seq_avg += duration;

        vector<atomic<int>> dist_par(g.size());
        for (int i = 1; i < g.size(); i++) {
            dist_par[i] = -1;
        }
        vector<int> deg(1e7);
        vector<int> mask(1e7);
        vector<int> tree(4e7);
        vector<int> frontier(1e7, -1);
        vector<int> new_frontier(1e7, -1);
        start = chrono::steady_clock::now();
        int ans_par = parallel_bfs(g, frontier, new_frontier, deg, mask, tree, dist_par);
        stop = chrono::steady_clock::now();
        duration = chrono::duration_cast<chrono::milliseconds>(stop - start).count();
        cout << "PAR: " << duration << " ms" << endl << endl;
        par_avg += duration;

        if (!equals(dist_seq, dist_par)) {
            cout << ans_seq << " " << ans_par << endl;
            throw exception();
        }
    }
    cout << "SEQ avg: " << seq_avg / 5 << endl;
    cout << "PAR avg: " << par_avg / 5 << endl;
    cout << "Score: " << seq_avg / par_avg;
}

int main() {
    bench();
    return 0;
}
