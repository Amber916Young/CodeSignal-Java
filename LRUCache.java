package HackerPage;

import java.util.HashMap;
import java.util.Map;

/**
 * @ClassName:LRUCache
 * @Auther: yyj
 * @Description:
 * @Date: 11/10/2022 16:49
 * @Version: v1.0
 */
public class LRUCache {
    final DoubleNode head = new DoubleNode();
    final DoubleNode tail = new DoubleNode();
    Map<Integer,DoubleNode> map = new HashMap<>();
    int cache_capacity = 0;
    public LRUCache(int capacity) {
        map = new HashMap<>(capacity);
        head.next = tail;
        tail.prev = head;
        this.cache_capacity = capacity;
    }

    public int get(int key) {
        int result = -1;
        DoubleNode node = map.get(key);
        if(node != null){
            result = node.val;
            remove(node);
            add(node);
        }
        return result;
    }

    public void put(int key, int value) {
        DoubleNode node = map.get(key);
        if(node != null){
            remove(node);
            node.val = value;
            add(node);
        }else {
            if(map.size() == cache_capacity){
                map.remove(tail.prev.key);
                remove(tail.prev);
            }
            DoubleNode newNode = new DoubleNode();
            newNode.key = key;
            newNode.val = value;
            map.put(key,newNode);
            add(newNode);
        }
    }

    public void add(DoubleNode node){
        DoubleNode next_node = head.next;
        node.next = next_node;
        next_node.prev = node;
        head.next = node;
        node.prev = head;

    }

    public void remove(DoubleNode node){
        DoubleNode next_node = node.next;
        DoubleNode prev_node = node.prev;
        next_node.prev = prev_node;
        prev_node.next = next_node;
     }
    class DoubleNode {
        int key;
        int val;
        DoubleNode prev;
        DoubleNode next;
    }
}
