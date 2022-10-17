import java.util.*;

/**
 * @ClassName:GraphLeetcode
 * @Auther: yyj
 * @Description:
 * @Date: 06/09/2022 12:43
 * @Version: v1.0
 */
public class GraphLeetcode {
    public static void main(String[] args) {
        GraphLeetcode leetcode = new GraphLeetcode();
        int[][] matrix2 ={
                { 1, 1, 1,-1,-1},
                { 1, -1, -1,-1,-1},
                {-1,-1,-1, 1, 1},
                { -1, -1, 1, 1,-1},
                {-1,-1,-1,-1,-1}
        };

        int[] ans = new int[]{1,15,7,9,2,5,10};
        List<List<Integer>> edges = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        list.add(2);
        list.add(3);
        edges.add(list);
//        list = new ArrayList<>();
//        list.add(1);
//        list.add(3);
//        edges.add(list);
//        list = new ArrayList<>();
//        list.add(3);
//        list.add(4);
//        edges.add(list);
        leetcode.bfs(3,1,edges,2);
    }

    public static List<Integer> bfs(int n, int m, List<List<Integer>> edges, int s) {
        int[] levelArray = new int[n+1];
        levelArray[0] = -1;
        levelArray[s] = -1;
        List<List<Integer>> graph = new ArrayList<>();
        for(int i=0;i<n+1;i++) graph.add(new ArrayList<>());
        for(List<Integer> edge: edges) {
            graph.get(edge.get(0)).add(edge.get(1));
            graph.get(edge.get(1)).add(edge.get(0));
        }
        Queue<Integer> queue = new ArrayDeque<>();
        queue.offer(s);
        int level = 0;
        while(!queue.isEmpty()){
            level++;
            int size = queue.size();
            for(int i=0;i<size;i++){
                int cur = queue.poll();
                for(int next: graph.get(cur)){
                    if(levelArray[next]==0){
                        levelArray[next] = level;
                        queue.offer(next);
                    }
                }
            }
        }
        List<Integer> answer = new ArrayList<>();
        for(int i=1;i<n+1;i++){
            if(i==s) continue;
            if(levelArray[i]!=0) answer.add(levelArray[i]*6);
            else answer.add(-1);
        }
        return answer;
    }


    public int concatenatedBinary(int n) {
        final long modulo = (long) (1e9 + 7);
        long result = 0;
        for (int i = 1; i <= n; i++) {
            int temp = i;
            while (temp > 0) {
                temp /= 2;
                result *= 2;
            }
            result = (result + i) % modulo;
        }
        return (int) result;
    }

    //TODO =============
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();

        boolean[] visited = new boolean[nums.length];
        Arrays.sort(nums);
        permuteUnique_DFS(nums, ans, new ArrayList<Integer>(),visited);
        return ans;
    }

    private void permuteUnique_DFS(int[] nums, List<List<Integer>> ans, ArrayList<Integer> list, boolean[] visited) {
        if (list.size() == nums.length) {
            ans.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if(  visited[i] ) continue; // 去重
            if(i > 0 && nums[i] == nums[i-1] && !visited[i - 1]) continue; // 前一个数 如果 等于后一个数 num[1] == num[0] == 2 and num[0]没有被访问过
            visited[i] = true;
            list.add(nums[i]);
            permuteUnique_DFS(nums, ans, list,visited);
            list.remove(list.size()-1);
            visited[i] = false;
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        permute_DFS(nums, ans, new ArrayList<Integer>(),visited);
        return ans;
    }

    private void permute_DFS(int[] nums, List<List<Integer>> ans, ArrayList<Integer> list, boolean[] visited) {
        if (list.size() == nums.length) {
            ans.add(new ArrayList<>(list));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if(  visited[i] )continue;
            visited[i] = true;
//            if(list.contains(nums[i])) continue; // element already exists, skip

            list.add(nums[i]);
            permute_DFS(nums, ans, list,visited);
            list.remove(list.size()-1);
            visited[i] = false;

        }
    }


    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        subsetsWithDup_DFS(nums,  ans, new ArrayList<Integer>(),0);
        return ans;
    }

    private void subsetsWithDup_DFS(int[] nums, List<List<Integer>> ans, ArrayList<Integer> list, int start) {
        ans.add(new ArrayList<>(list));
        for(int i = start;i<nums.length;i++){
            if(i > start && nums[i] == nums[i-1]) continue;// skip duplicates
            list.add(nums[i]);
            subsetsWithDup_DFS(nums,  ans,list ,i+1);
            list.remove(list.size()-1);
        }
    }



    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(candidates);
        combinationSum_DFS(candidates, 0, target, ans, new ArrayList<Integer>(),0);
        return ans;
    }

    private void combinationSum_DFS(int[] candidates, int sum, int target, List<List<Integer>> ans,
                                    ArrayList<Integer> list, int start) {
        if(sum > target) return;
        if(sum == target){
            ans.add(new ArrayList<>(list));
            return;
        }
        for(int i = start;i<candidates.length;i++){
            list.add(candidates[i]);
            combinationSum_DFS(candidates, sum + candidates[i], target, ans, list,i);
            list.remove(list.size()-1);
        }
    }




    public List<List<Integer>> combinationSum2(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<>();
        Arrays.sort(nums);
        combinationSum2_DFS(nums, 0, target, ans, new ArrayList<Integer>(),0);
        return ans;
    }

    private void combinationSum2_DFS(int[] candidates, int sum, int target, List<List<Integer>> ans,
                                    ArrayList<Integer> list, int start) {
        if(sum > target) return;
        if(sum == target){
            ans.add(new ArrayList<>(list));
            return;
        }
        for(int i = start;i<candidates.length;i++){
            if(i > start && candidates[i] == candidates[i-1]) continue; // skip duplicates
            list.add(candidates[i]);
            combinationSum2_DFS(candidates, sum + candidates[i], target, ans, list,i);
            list.remove(list.size()-1);
        }
    }



    //131. Palindrome Partitioning
    /**
     *
     *
     * Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
     * A palindrome string is a string that reads the same backward as forward.
     * Example 1:
     * Input: s = "aab"
     * Output: [["a","a","b"],["aa","b"]]
     *
     * Example 2:
     * Input: s = "a"
     * Output: [["a"]]
     *
     * **/
    public List<List<String>> partition(String s) {
        List<List<String>> ans = new ArrayList<>();
        partition_DFS(s, ans, new ArrayList<String>(),0);
        return ans;
    }
    public boolean isPalindrome(String s, int low, int high){
        while(low < high)
            if(s.charAt(low++) != s.charAt(high--)) return false;
        return true;
    }
    private void partition_DFS(String s, List<List<String>> ans,
                                     ArrayList<String> list, int start) {
        if(start == s.length()){
            ans.add(new ArrayList<>(list));
            return;
        }
        for(int i = start;i<s.length();i++){
            if (isPalindrome(s, start, i)) {
                list.add(s.substring(start, i + 1));
                partition_DFS(s, ans, list,i+1);
                list.remove(list.size()-1);
            }
        }
    }

    //TODO =============


    public int reachableNodes(int n, int[][] edges, int[] restricted) {
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for (int[] g : edges) {
            int source = g[1];
            int dest = g[0];
            graph[source].add(dest);
            graph[dest].add(source);
        }

        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> rest = new HashSet<>();
        for (int i : restricted) rest.add(i);
        int ans = 0;
        queue.offer(0);
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (rest.contains(cur)) continue;
            rest.add(cur);
            for (int tmp : graph[cur]) {
                if (!rest.contains(tmp) ) {
                    queue.offer(tmp);
                }
            }
            ans++;
        }
        return ans;
    }



    public int[] findBall(int[][] grid) {
        int col = grid[0].length;
        int[] ans = new int[col];
        for(int i = 0;i<col;i++){
            ans[i] =  DFS_findBall(grid,i,0);
        }
        return ans;
    }
    private int DFS_findBall(int[][] grid, int col,int row) {
        if (row == grid.length) {
            return col;
        }
        int nextColumn = col + grid[row][col];
        if (nextColumn < 0 || nextColumn > grid[0].length - 1 || grid[row][col] != grid[row][nextColumn]) {
            return -1;
        }
        return DFS_findBall(grid,nextColumn, row + 1);
    }

    public int minStoneSum(int[] piles, int k) {

        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((a,b)->(b-a));
        for(int value: piles){
            priorityQueue.offer(value);
        }
        while (k>0){
            int pop = priorityQueue.poll();
            int tmp =pop%2==0 ? pop/2 : (pop/2 +1) ;
            priorityQueue.add(tmp);
            k--;
        }
        int ans = 0 ;
        while (!priorityQueue.isEmpty()){
            ans += priorityQueue.poll();
        }
        return ans;
    }

    public int minSetSize(int[] arr) {
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int a:arr){
            map.put(a,map.getOrDefault(a,0)+1);
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((a,b)->(b-a));
        for(int value:map.values()){
            priorityQueue.offer(value);
        }
        int sum = 0;
        int count = 0;
        while(sum < arr.length/2){
            sum+= priorityQueue.poll();
            count++;
        }
        return count;
    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        int[] cost = new int[n];
        Arrays.fill(cost, Integer.MAX_VALUE);
        cost[src] = 0;
        for (int i = 0; i <= k; i++) {
            int[] temp = Arrays.copyOf(cost, n);
            for (int[] f : flights) {
                int curr = f[0], next = f[1], price = f[2];
                if (cost[curr] == Integer.MAX_VALUE)
                    continue;
                temp[next] = Math.min(temp[next], cost[curr] + price);
            }
            cost = temp;
        }
        return cost[dst] == Integer.MAX_VALUE ? -1 : cost[dst];
    }


    public int maxSumAfterPartitioning(int[] arr, int K) {
        int N = arr.length, dp[] = new int[N];
        for (int i = 0; i < N; ++i) {
            int curMax = 0;
            for (int k = 1; k <= K && i - k + 1 >= 0; ++k) {
                System.out.println(k);
                System.out.println(i - k + 1);
                curMax = Math.max(curMax, arr[i - k + 1]);
                dp[i] = Math.max(dp[i], (i >= k ? dp[i - k] : 0) + curMax * k);
            }
        }
        return dp[N - 1];
    }

    //1282. Group the People Given the Group Size They Belong To
    public List<List<Integer>> groupThePeople(int[] groupSizes) {
        List<List<Integer>> ans = new ArrayList<>();
        HashMap<Integer,List<Integer>> map = new HashMap<>();
        for(int i =0;i<groupSizes.length;i++) {
            int temp = groupSizes[i];
            if (map.containsKey(temp)) {
                List<Integer> list = map.get(temp);
                if (list.size() == temp) {
                    ans.add(list);
                    list = new ArrayList<>();
                    map.put(temp, list);
                }
                list.add(i);
                map.put(temp, list);

            } else {
                List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(temp, list);
            }
        }
        Iterator < Integer > iterator = map.keySet().iterator();
        while (iterator.hasNext()){
            Integer key = iterator.next();
            ans.add( map.get(key));
        }
        return ans;
    }


    //Bellman Ford Algorithm:
    public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
        List<double[]>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }
        for(int i=0;i<edges.length;i++) {
            graph[edges[i][0]].add(new double[]{edges[i][1], succProb[i]});
            graph[edges[i][1]].add(new double[]{edges[i][0], succProb[i]});
        }
        Queue<double[]> MaxHeap = new PriorityQueue<>(Collections.reverseOrder((a, b) -> Double.compare(a[1], b[1])));
        MaxHeap.offer(new double[]{start,1.0});
        double max  =  0.0;
        HashSet<Integer> visited = new HashSet<>();
        double[] probabilities = new double[n];
        while (!MaxHeap.isEmpty()){
            double[] cur = MaxHeap.poll();
            int point = (int) cur[0];
            double currentProbability = cur[1];
            if(visited.contains(point)) continue;
            visited.add(point);
            for(double[] gedges : graph[point]){
                int next = (int) gedges[0];
                double probability = gedges[1];
                if(probabilities[next] < currentProbability * probability) {
                    probabilities[next] = currentProbability * probability;
                    MaxHeap.add(new double[]{next, probabilities[next]});
                }
            }
        }
        return max;
    }


    public int[] findOrder(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> adjList = new HashMap<Integer, List<Integer>>();
        int[] indegree = new int[numCourses];
        int[] topologicalOrder = new int[numCourses];

        for (int i = 0; i < prerequisites.length; i++) {
            int dest = prerequisites[i][0];
            int src = prerequisites[i][1];
            List<Integer> lst = adjList.getOrDefault(src, new ArrayList<Integer>());
            lst.add(dest);
            adjList.put(src, lst);
            indegree[dest]++;
        }
        Queue<Integer> q = new LinkedList<Integer>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) q.add(i);
        }
        int i = 0;
        while (!q.isEmpty()) {
            int node = q.remove();
            topologicalOrder[i++] = node;
            if (adjList.containsKey(node)) {
                for (Integer neighbor : adjList.get(node)) {
                    indegree[neighbor]--;
                    if (indegree[neighbor] == 0) {
                        q.add(neighbor);
                    }
                }
            }
        }
        if (i == numCourses) {
            return topologicalOrder;
        }

        return new int[0];
    }
    public int[] findOrder_2(int numCourses, int[][] prerequisites) {
        int[] res = new int[numCourses];
        int[] inDegree = new int[numCourses];
        Map<Integer,List<Integer>> graph = new HashMap<>();
        Queue<Integer> queue = new LinkedList<>();
        List<Integer> order = new ArrayList<>();
        for(int[] p : prerequisites){
            int pre = p[1];
            int course = p[0];
            inDegree[course]++;
            if (!graph.containsKey(pre)) graph.put(pre, new ArrayList<>());
            graph.get(pre).add(course);
        }
        for (int i = 0; i < numCourses; i++) {
            if(inDegree[i] == 0){
                queue.add(i);
                order.add(i);
            }
        }
        if(queue.isEmpty()) return new int[]{};
        while (!queue.isEmpty()){
            int cur = queue.poll();
            List<Integer> courses = graph.get(cur);
            if(courses == null) continue;
            for(Integer c : courses){
                inDegree[c]--;
                if(inDegree[c]==0) {
                    queue.add(c);
                    if(order.contains(c)) continue;
                    order.add(c);
                }

            }
        }
        for(int i =0;i<order.size() && order.size() == numCourses;i++) {
            res[i] = order.get(i);
        }
        return res;
    }

    public boolean canFinish_BFS(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        Map<Integer,List<Integer>> graph = new HashMap<>();
//        for (int i = 0; i < numCourses; i++) {
//            graph.put(i, new ArrayList<Integer>());
//        }
        for(int[] c : prerequisites){
            int next = c[0];
            int pre = c[1];
            inDegree[next]++;
            if(graph.containsKey(pre)){
                graph.get(pre).add(next);
            }else {
                ArrayList<Integer> list = new ArrayList<>();
                list.add(next);
                graph.put(pre,list);
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if(inDegree[i] == 0)queue.add(i);
        }
        int count = 0;
        while (!queue.isEmpty()){
            int cur = queue.poll();
            if(graph.get(cur) != null) {
                for(int i : graph.get(cur)){
                    inDegree[i]--;
                    if(inDegree[i]==0)queue.add(i);
                }
            }
            count++;
        }

        return count==numCourses;
    }


    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<Integer>[] graph = new ArrayList[numCourses];
        for (int i = 0; i < numCourses; i++) {
            graph[i] = new ArrayList<>();
        }
        for(int[] g : prerequisites) {
            int source = g[1];
            int dest = g[0];
            graph[source].add(dest);
        }
        boolean[] memo = new boolean[numCourses];

        for(int i = 0;i <numCourses;i++){

            boolean[] visited = new boolean[numCourses];
            if(!canFinish_DFS(visited,graph,i,memo)){
                return false;
            }
        }
        return true;
    }

    private boolean canFinish_DFS(boolean[] visited, List<Integer>[] graph, int next,boolean[] memo) {
        if(visited[next]) return false;
        if(memo[next]) return true;
        visited[next] = true;
        for(int nextnode : graph[next]){
            if(!canFinish_DFS(visited,graph,nextnode,memo)) return false;
        }
        memo[next] = true;
        return true;
    }


    //210. Course Schedule II




    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length;
        List<Integer>[] graph = new ArrayList[n];
        HashMap<Integer, Integer> loudMap = new HashMap<>();
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
            loudMap.put(i,quiet[i]);
            map.put(quiet[i],i);
        }
        for(int[] g : richer){
            int source = g[1];
            int dest = g[0];
            graph[source].add(dest);
        }
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            boolean[] visited = new boolean[n];
            minLoud = Integer.MAX_VALUE;
            loudAndRich_DFS(graph,i,loudMap,visited);
            ans[i] = map.get(minLoud) ;
        }
        return ans;
    }

    int minLoud = Integer.MAX_VALUE;
    private void loudAndRich_DFS(List<Integer>[] graph, int next, HashMap<Integer, Integer> loudMap, boolean[] visited ) {
        if (visited[next]) return;
        visited[next] = true;
        int loud =  loudMap.get(next);
        minLoud = Math.min(minLoud, loud);
        for(int nextNode : graph[next]){
            loudAndRich_DFS(graph,nextNode,loudMap,visited);
        }
    }


    public int[] sumEvenAfterQueries(int[] nums, int[][] queries) {
        int count = 0;
        for (int j = 0; j < nums.length; j++) {
            if(nums[j] % 2 == 0) count +=nums[j];
        }
        int[] ans = new int[nums.length];
        for (int i = 0;i<nums.length;i++) {
            int val = queries[i][0];
            int index = queries[i][1];
            int num = nums[index];
            nums[index] = num + val;
            if ((nums[index] % 2 == 0 && num % 2 != 0) ) {
                count = count + nums[index];
                ans[i] = count;
            }else if(nums[index] % 2 != 0 && num % 2 == 0){
                count = count - num;
                ans[i] = count;
            }else if(nums[index] % 2 == 0 && num % 2 == 0){
                count = count  + val;
                ans[i] = count;
            }else {
                ans[i] = count;
            }
        }
        return ans;
    }


    public long maximumImportance(int n, int[][] roads) {
        long ans = 0, x = 1;
        long degree[] = new long[n];
        for(int road[] : roads){
            degree[road[0]]++;
            degree[road[1]]++;
        }
        Arrays.sort(degree);
        for(long node : degree){
            ans +=  node * (x++) ;
        }
        return ans;
    }

    public int maximalNetworkRank(int n, int[][] roads) {
        boolean visited[][] = new boolean[n][n];
        int rds[] = new int[n];
        for(int[] road : roads) {
            rds[road[0]]++;
            rds[road[1]]++;
            visited[road[0]][road[1]] = true;
            visited[road[1]][road[0]] = true;
        }
        int ans = 0;
        for(int i=0; i<n; i++) {
            for (int j = i + 1; j < n; j++) {
                int tmp = rds[i] + rds[j];
                if (visited[i][j]) {
                    tmp = tmp - 1;
                }
                ans = Math.max(ans, tmp);
            }
        }
        return ans;
    }


    public int singleNonDuplicate(int[] nums) {
       return 0;
    }

    class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }
    public romanToInt.TreeNode createBinaryTree(int[][] descriptions) {
        //[[20,15,1],[20,17,0],[50,20,1],[50,80,0],
        HashMap<Integer, romanToInt.TreeNode> x = new HashMap<>();
        HashSet<romanToInt.TreeNode> par = new HashSet<>();
        for(int [] arr : descriptions){
            if(!x.containsKey(arr[0])){
                x.put(arr[0], new romanToInt.TreeNode(arr[0]));
                par.add(x.get(arr[0]));
            }
            romanToInt.TreeNode curr = x.get(arr[0]);

            if(!x.containsKey(arr[1]))
                x.put(arr[1], new romanToInt.TreeNode(arr[1]));
            else par.remove(x.get(arr[1]));

            romanToInt.TreeNode add = x.get(arr[1]);
            if(arr[2]==1) curr.left = add;
            else curr.right = add;
        }
        return par.iterator().next();
    }
    public Node cloneGraph(Node node) {
        return cloneGraphDFSHelper(node, new HashMap<>());
    }
    private Node cloneGraphDFSHelper(Node cur, HashMap<Node, Node> visited) {
        if (cur == null) {
            return null;
        }
        if (visited.containsKey(cur)) {
            return visited.get(cur);
        }
        Node newNode = new Node(cur.val);
        visited.put(cur, newNode);
        for (Node n : cur.neighbors) {
            newNode.neighbors.add(cloneGraphDFSHelper(n, visited));
        }

        return newNode;
    }
    public Node cloneGraph2(Node node) {
        if (node == null) {
            return null;
        }
        HashMap<Node, Node> visited = new HashMap<>();
        Node newNode = new Node(node.val);
        visited.put(node, newNode);
        Queue<Node> queue = new LinkedList<>();
        queue.offer(node);
        while (!queue.isEmpty()) {
            Node cur = queue.poll();
            List<Node> newNeighbors = visited.get(cur).neighbors;
            for (Node n : cur.neighbors) {
                if (!visited.containsKey(n)) {
                    visited.put(n, new Node(n.val));
                    queue.offer(n);
                }
                newNeighbors.add(visited.get(n));
            }
        }

        return newNode;
    }


    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        return new int[]{};
    }



    public int findCircleNum(int[][] isConnected) {
        int N = isConnected.length;
        boolean[]visited = new boolean[N];
        int count = 0;
        for(int i = 0; i < N ;i++){
            if(!visited[i]){
                count++;
                findCircleNum_dfs(isConnected,i,visited);
            }
        }
        return count;
    }
    private void findCircleNum_dfs(int[][]isConnected,int i,boolean[]visited){
        for(int j = 0 ; j < isConnected.length ; j++){
            if(!visited[j] && isConnected[i][j] != 0){
                visited[j] = true;
                findCircleNum_dfs(isConnected,j,visited);
            }
        }
    }

    public int minReorder(int n, int[][] connections) {
        List<List<Integer>> graphs = new ArrayList<>();
        for(int i =0;i<n;i++){
            graphs.add(new ArrayList<Integer>());
        }
        for(int[] tmp : connections){
            graphs.get(tmp[0]).add(tmp[1]);
            graphs.get(tmp[1]).add(-tmp[0]);
        }
        return minReorder_DFS(graphs,new boolean[n],0);
    }

    private int minReorder_DFS(List<List<Integer>> graphs, boolean[] visited,int root) {
        int count = 0;
        visited[root] = true;
        for(int to : graphs.get(root)){
            if(!visited[Math.abs(to)]) {
                count += minReorder_DFS(graphs, visited, Math.abs(to)) + ((to > 0) ? 1 : 0);
            }
        }
        return count;
    }

    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] time : times) {
            int source = time[0];
            int dest = time[1];
            int cost = time[2];
            if (!graph.containsKey(source)) {
                graph.put(source, new LinkedList<int[]>());
            }
            graph.get(source).add(new int[]{dest, cost});
        }
        // sort by cost(weight)
        Queue<int[]> queue = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        Set<Integer> visited = new HashSet<>();
        queue.add(new int[]{k, 0});
        int res = 0;
        while (!queue.isEmpty()) {
            int[] top = queue.poll();
            int source = top[0];
            int cost = top[1];
            if (visited.contains(source)) continue;
            res = cost;
            visited.add(source);
            if (!graph.containsKey(source)) continue;
            for (int[] edge : graph.get(source)) {
                int dest = edge[0];
                int dest_cost = edge[1];
                queue.offer(new int[]{dest, cost + dest_cost});
            }
        }
        return visited.size() == n ? res : -1;
    }


    //1654. Minimum Jumps to Reach Home
    public int minimumJumps(int[] forbidden, int a, int b, int x) {
        HashSet<String> visited = new HashSet<>();
        HashSet<Integer> forbiddenList = new HashSet<Integer>();
        for (int i : forbidden) forbiddenList.add(i);
        Queue<int[]> queue = new LinkedList<>();
        int count = 0;
        int max = 2000;
        queue.offer(new int[]{0, 0});
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                int pos = cur[0], direction = cur[1];
                if(pos == x) return count;
                if(pos + a < max && !forbiddenList.contains(pos + a) && !visited.contains(pos+a+","+0)){
                    visited.add(pos+a+","+0);
                    queue.offer(new int[]{pos + a, 0});
                }
                if(direction == 0) {
                    if(pos - b >= 0 && !forbiddenList.contains(pos - b) && !visited.contains(pos-b+","+1)) {
                        visited.add(pos-b+","+1);
                        queue.offer(new int[]{pos - b, 1});
                    }
                }
            }
            count++;
        }
        return -1;
    }
    // jump3
    public boolean canReach(int[] arr, int start) {
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        HashSet<Integer> visited = new HashSet<>();
        while (!queue.isEmpty()){
            int size = queue.size();
            for(int i =0 ;i<size;i++) {
                int curr = queue.poll();
                if(!visited.contains(curr)){
                    visited.add(curr);
                }else continue;
                if(curr-arr[curr] >= 0){
                    if(arr[curr-arr[curr]] == 0) return true;
                    queue.offer(curr-arr[curr]);
                }
                if(curr+arr[curr] < arr.length){
                    if(arr[curr+arr[curr]] == 0) return true;
                    queue.offer(curr+arr[curr]);
                }
            }
        }
        return false;
    }




    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        List<Integer>[] map = new ArrayList[n];
        for (int i = 0; i < n; i++) map[i] = new ArrayList<>();
        for (int[] c : graph) {
            map[c[0]].add(c[1]);
            map[c[1]].add(c[0]);
        }

        return 0;


    }


    public List<Integer> findSmallestSetOfVertices(int n, List<List<Integer>> edges) {
        List<Integer> res = new ArrayList<>();
        int[] seen = new int[n];
        for (List<Integer> e: edges)
            seen[e.get(1)] = 1;
        for (int i = 0; i < n; ++i)
            if (seen[i] == 0)
                res.add(i);
        return res;
    }

    public int makeConnected(int n, int[][] connections) {
        int edge = n - 1;
        if (connections.length < edge) return -1;
        List<List<Integer>> graphs = new ArrayList<>();
        for(int i =0;i<n;i++){
            graphs.add(new ArrayList<Integer>());
        }
        for(int[] tmp : connections){
            graphs.get(tmp[0]).add(tmp[1]);
            graphs.get(tmp[1]).add(tmp[0]);
        }
        boolean[] visited = new boolean[n];
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += makeConnected_DFS(graphs, visited, i);
        }
        return ans - 1;
    }
    private int makeConnected_DFS( List<List<Integer>> graphs, boolean[] visited,int next) {
        if (visited[next]) return 0;
        visited[next] = true;
        for(int nextNode : graphs.get(next)){
            makeConnected_DFS(graphs,visited,nextNode);
        }
        return 1;
    }



    public int findCenter(int[][] edges) {
//        Map<Integer, Integer> map = new HashMap<>();
//        for(int[] edge : edges){
//            map.put(edge[0], map.getOrDefault(edge[0], 0)+ 1);
//            map.put(edge[1], map.getOrDefault(edge[1], 0)+ 1);
//        }
//        int max = 0, res = 0;
//        for(int key : map.keySet()){
//            if(map.get(key) > max){
//                max = map.get(key);
//                res = key;
//            }
//        }
//        return res;

        int [] count = new int[edges.length+2];

        for(int i=0;i<edges.length;i++) {
            count[edges[i][0]]++;
            count[edges[i][1]]++;
        }

        for(int i=1;i<count.length; i++) {
            if(count[i]==edges.length) return i;
        }
        return -1;
    }




    //BFS
    public int makeConnectedBFS(int n, int[][] connections) {
        if (connections.length < n - 1) return -1;
        List<Integer>[] graph = new List[n];
        for (int i = 0; i < n; i++) graph[i] = new ArrayList<>();
        for (int[] c : connections) {
            graph[c[0]].add(c[1]);
            graph[c[1]].add(c[0]);
        }
        int components = 0;
        boolean[] visited = new boolean[n];
        for (int v = 0; v < n; v++) components += bfs(v, graph, visited);
        return components - 1;
    }
    int bfs(int src, List<Integer>[] graph, boolean[] visited) {
        if (visited[src]) return 0;
        visited[src] = true;
        Queue<Integer> q = new LinkedList<>();
        q.offer(src);
        while (!q.isEmpty()) {
            int u = q.poll();
            for (int v : graph[u]) {
                if (!visited[v]) {
                    q.offer(v);
                }
                visited[v] = true;
            }
        }
        return 1;
    }

    //997. Find the Town Judge
    public int findJudge(int n, int[][] trust) {
        int[] count = new int[n+1];
        for (int[] t: trust) {
            count[t[0]]--;
            count[t[1]]++;
        }
        for (int i = 1; i <= n; ++i) {
            if (count[i] == n - 1) return i;
        }
        return -1;
    }

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        path.add(0);
        allPathsSourceTarget_DFS(graph,0,path,ans);
        return ans;
    }

    private void allPathsSourceTarget_DFS(int[][] graph,int nextNode, List<Integer> path, List<List<Integer>> ans) {
        if(nextNode == graph.length-1 ){
            ans.add(new ArrayList<Integer>(path));
            return;
        }
        for (int next : graph[nextNode]){
            path.add(next);
            allPathsSourceTarget_DFS(graph,next,path,ans);
            path.remove(path.size()-1);
        }
    }



}
