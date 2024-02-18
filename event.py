import topology
import numpy as np
from rl import Model, ConvDense, Dense
import tensorflow as tf

def build_timeline(num_req, path, start):
    timeline = []
    for i in range(start, start + num_req):
        req = topology.build_virtual_request(i, path)
        timeline.append((req.t_begin, req))
        timeline.append((-req.t_end, req))
    timeline.sort(key=lambda e: abs(e[0]))
    return timeline


def run_base(sub_path, req_path, num_req, start):
    timeline = build_timeline(num_req, req_path, start)
    start_time, end_time = (timeline[0][0], -timeline[-1][0])
    network = topology.build_substrate_network(sub_path)
    total_revenue = total_cost = accepted = current_req_sum = 0.0
    write_time = 2000 + start_time
    with open("base_" + str(start) + "_" + str(start + num_req), "w") as output:
        for event in timeline:
            if abs(event[0]) > write_time:
                output.write(str(total_revenue/(write_time - start_time)))
                output.write("\t")
                output.write(str(accepted / current_req_sum))
                output.write("\t")
                output.write(str(total_revenue / total_cost))
                output.write("\n")
                write_time += 2000
            req = event[1]
            if event[0] < 0:
                if req.mapped == True:
                    req.unmap(network)
                    req.mapped = False
            else:
                current_req_sum += 1
                status = req.node_map(network)
                if not status:
                    req.unmap(network)
                else:
                    status = req.link_map(network)
                    if not status:
                        req.unmap(network)
                    else:
                        for link in req.virtual_links:
                            total_cost += link.cost * (req.t_end - req.t_begin)
                        total_revenue += req.revenue
                        accepted += 1
                        req.mapped = True
                        #print("req %d mapped" % req.index)
        print(total_revenue/(end_time - start_time))
        print(accepted / num_req)
        print(total_revenue / total_cost)
        output.write(str(total_revenue / (end_time - start_time)))
        output.write("\t")
        output.write(str(accepted / num_req))
        output.write("\t")
        output.write(str(total_revenue / total_cost))
        output.write("\n")


def run_rl(sub_path, req_path, num_req, start, num_epoch=50, target='rc', test=False, model_str='conv', learning_rate=0.005):
    output = False
    test_num = 0
    timeline = build_timeline(num_req, req_path, start)
    start_time, end_time = (timeline[0][0], -timeline[-1][0])
    network = topology.build_substrate_network(sub_path)
    if model_str == 'conv':
        model = Model()
    if model_str == 'dense':
        model = Dense()
    if model_str == 'convdense':
        model = ConvDense()
    print('using ' + model_str) 
    if target == 'rev':
        best_res = 100.0
    elif target == 'acc':
        best_res = 0.65
    elif target == 'rc':
        best_res = 0.33
    for epoch in range(num_epoch):
        total_revenue = total_cost = accepted = current_req_sum = update_counter = 0.0
        reward = []
        for event in timeline:
            req = event[1]
            if event[0] < 0:
                # 结束释放
                if req.mapped == True:
                    req.unmap(network)
                    req.mapped = False
            else:
                current_req_sum += 1
                status = req.node_map_rl(network, model)
                if not status:
                    req.unmap(network)
                else:
                    status = req.link_map(network)
                    if not status:
                        req.unmap(network)
                        model.del_grad(len(req.virtual_nodes))
                    else:
                        cost = 0.0
                        for link in req.virtual_links:
                            cost += link.cost
                        cost *= (req.t_end - req.t_begin)
                        r = (req.revenue / cost - 0.35) * learning_rate 
                        reward.extend([r] * len(req.virtual_nodes))
                        if len(reward) != len(model.grads):
                            print(len(reward), len(model.grads))
                        if current_req_sum - update_counter >= 200:
                            update_counter += 100
                            model.update_grads(reward)
                            reward = []
                        total_revenue += req.revenue
                        total_cost += cost
                        accepted += 1
                        req.mapped = True
                #if current_req_sum % 500 == 0:
                    #for tvar in tf.trainable_variables():
                    #    print(np.reshape(model.sess.run(tvar), [-1]))
        model.update_grads(reward)
        rev = total_revenue/(end_time - start_time)
        acc = accepted / num_req
        rc = total_revenue / total_cost
        if target == 'rev':
            res = rev
        elif target == 'acc':
            res = acc
        elif target == 'rc':
            res = rc
        print('------------------------------------------------------------')
        print(rev, acc, rc)
        print('------------------------------------------------------------')
        for tvar in tf.trainable_variables():
            print(tvar.name)
            print(np.reshape(model.sess.run(tvar), [-1])[:10])
        #model.update_grads([reward])
        if test:
            if res > best_res:
                best_res = res
                output = True
        if output:
            test_num += 1
            timeline_test = build_timeline(2000 - num_req, req_path, start + num_req)
            #timeline_test = build_timeline(num_req, req_path, start)
            start_time, end_time = (timeline_test[0][0], -timeline_test[-1][0])
            total_revenue = total_cost = accepted = current_req_sum = 0.0
            write_time = 2000 + start_time
            with open('_'.join(['rl', str(start+num_req), str(2000), str(test_num)]), 'w') as output:
                for event in timeline_test:
                    if abs(event[0]) > write_time:
                        output.write(str(total_revenue/(write_time - start_time)))
                        output.write("\t")
                        output.write(str(accepted / current_req_sum))
                        output.write("\t")
                        output.write(str(total_revenue / total_cost))
                        output.write("\n")
                        write_time += 2000
                    req = event[1]
                    if event[0] < 0:
                        if req.mapped == True:
                            req.unmap(network)
                            req.mapped = False
                    else:
                        current_req_sum += 1
                        status = req.node_map_rl(network, model, test=True)
                        if not status:
                            req.unmap(network)
                        else:
                            status = req.link_map(network)
                            if not status:
                                req.unmap(network)
                            else:
                                for link in req.virtual_links:
                                    total_cost += link.cost * (req.t_end - req.t_begin)
                                total_revenue += req.revenue
                                accepted += 1
                                req.mapped = True

                rev = total_revenue/(end_time - start_time)
                acc = accepted / num_req
                rc = total_revenue / total_cost
                if target == 'rev':
                    res = rev
                elif target == 'acc':
                    res = acc
                elif target == 'rc':
                    res = rc
                print(rev, acc, rc)
                output.write(str(total_revenue / (end_time - start_time)))
                output.write("\t")
                output.write(str(accepted / num_req))
                output.write("\t")
                output.write(str(total_revenue / total_cost))
                output.write("\n")
                output = False
