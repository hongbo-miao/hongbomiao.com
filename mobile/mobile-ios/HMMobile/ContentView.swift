//
//  ContentView.swift
//  HMMobile
//
//  Created by Hongbo Miao on 11/21/22.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Text("HONGBO MIAO")
                .font(.largeTitle)
                .fontWeight(.black)
            Text("Making magic happen")
                .font(.title3)
                .padding(.top)
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
