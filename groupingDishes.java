package HashTable;

import java.util.*;

/**
 * @ClassName:HashTables
 * @Auther: yyj
 * @Description:
 * @Date: 17/10/2022 11:37
 * @Version: v1.0
 */
public class groupingDishes {
//   new String[][]{
//        {"Salad", "Tomato", "Cucumber", "Salad", "Sauce"}
//                ,{"Pizza", "Tomato", "Sausage", "Sauce", "Dough"}
//                ,{"Quesadilla", "Chicken", "Cheese", "Sauce"}
//                ,{"Sandwich", "Salad", "Bread", "Tomato", "Cheese"}
//    };
    String[][] groupingDishes(String[][] dishes) {
        HashMap<String, List<String>> map= new HashMap<>();
        for(int i=0; i< dishes.length; i++){
            for(int  j=1; j< dishes[i].length; j++){
                List<String> list;
                if(!map.containsKey(dishes[i][j])){
                    list = new ArrayList<String>();

                }else{
                    list = map.get(dishes[i][j]);
                }
                list.add(dishes[i][0]);
                map.put(dishes[i][j], list);

            }
        }
        TreeMap<String, List<String>> treemap = new TreeMap<>(map);
        List<List<String>> reList= new ArrayList<>();
        for(String key: treemap.keySet()){
            if(treemap.get(key).size()>=2){
                List<String> subList= treemap.get(key);
                Collections.sort(subList);
                subList.add(0,key);
                reList.add(subList);
            }
        }

        System.out.println(reList.toString());
        String[][] array = new String[reList.size()][];
        for (int i = 0; i < reList.size(); i++) {
            List<String> row = reList.get(i);
            array[i] = row.toArray(new String[row.size()]);
        }

        return array;
    }


}
